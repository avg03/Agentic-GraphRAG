"""Supervisor (v2 §10.2/§13): routes the first action and every next action.

Action space (v2 §12): GRAPH_SEARCH (local), GRAPH_SCOPE (sibling
discovery), VECTOR_SEARCH, COMMUNITY_SEARCH, REFORMULATE, STOP — the
community proxy is retained from v1. Budget is enforced BEFORE the LLM call;
stop reason_codes follow the v2 §19 taxonomy.
"""

from shared.llm_client import generate_structured
from shared.models import SupervisorDecision
from shared.prompts import SUPERVISOR_SYSTEM_PROMPT, build_supervisor_prompt

__all__ = ["decide_next_action", "STOP_REASONS"]

# Hard mapping guard: tools we can still execute under budget.
_VALID_ACTIONS = {
    "GRAPH_SEARCH", "GRAPH_SCOPE", "VECTOR_SEARCH",
    "COMMUNITY_SEARCH", "REFORMULATE", "STOP",
}

# v2 §19: every stop must record WHY, from this closed set.
STOP_REASONS = {
    "sufficient_evidence",
    "diminishing_returns",
    "budget_exhausted",
    "no_useful_path",
}


def _normalize_stop_reason(reason_code: str) -> str:
    if reason_code in STOP_REASONS:
        return reason_code
    synonyms = {
        "all_slots_supported": "sufficient_evidence",
        "all_required_slots_supported": "sufficient_evidence",
        "answered": "sufficient_evidence",
        "max_steps": "budget_exhausted",
        "duplicate_scope": "diminishing_returns",
        "no_new_evidence": "diminishing_returns",
    }
    if reason_code in synonyms:
        return synonyms[reason_code]
    if "sufficient" in reason_code or "supported" in reason_code:
        return "sufficient_evidence"
    if "budget" in reason_code or "steps" in reason_code:
        return "budget_exhausted"
    if "diminish" in reason_code or "duplicate" in reason_code:
        return "diminishing_returns"
    return "no_useful_path"


def decide_next_action(
    question: str,
    slots: list,
    slot_statuses: dict,
    history: list,
    budget: dict,
    metrics=None,
    question_type: str = "",
    candidate_contexts: list | None = None,
    investigated_contexts: list | None = None,
    evidence_stats: dict | None = None,
) -> tuple[SupervisorDecision, dict]:
    """Decide the single next action. Budget is enforced BEFORE the LLM call."""
    # Budget enforcement (v2 §19 hard budget): no LLM call needed to know.
    if budget["steps"] >= budget["max_steps"]:
        return (
            SupervisorDecision(
                action="STOP",
                reason_code="budget_exhausted",
                note=f"step budget {budget['steps']}/{budget['max_steps']} used",
            ),
            {},
        )

    decision, usage = generate_structured(
        prompt=build_supervisor_prompt(
            question,
            slots,
            list(slot_statuses.values()),
            history,
            budget,
            question_type=question_type,
            candidate_contexts=candidate_contexts,
            investigated_contexts=investigated_contexts,
            evidence_stats=evidence_stats,
        ),
        response_schema=SupervisorDecision,
        system=SUPERVISOR_SYSTEM_PROMPT,
    )
    if metrics is not None:
        latency = usage.pop("_latency_ms", None)
        metrics.record_llm(usage, latency_ms=latency)
    elif usage:
        usage.pop("_latency_ms", None)

    if decision.action not in _VALID_ACTIONS:
        decision.action = "STOP"
        decision.reason_code = "no_useful_path"
        decision.note = "model proposed unknown action; forcing stop"

    if decision.action == "STOP":
        decision.reason_code = _normalize_stop_reason(decision.reason_code)
        return decision, usage or {}

    # Never re-investigate an already-investigated context root (v2 §11).
    if decision.action == "GRAPH_SCOPE":
        investigated = set(investigated_contexts or [])
        candidates = [c for c in (candidate_contexts or []) if c.get("key") not in investigated]
        requested = decision.target_context
        if requested in investigated or not any(c.get("key") == requested for c in candidates):
            remaining = [c["key"] for c in candidates]
            if remaining:
                decision.target_context = remaining[0]
                decision.note = (
                    f"target_context {requested or '(none)'} already investigated/unknown; "
                    f"redirected to {decision.target_context}"
                )
            else:
                decision.action = "STOP"
                decision.reason_code = "diminishing_returns"
                decision.note = "all candidate context roots already investigated"
                return decision, usage or {}

    # Per-tool budget: fall back to another affordable tool or STOP.
    caps = {
        "GRAPH_SEARCH": ("graph_calls", "max_graph"),
        "GRAPH_SCOPE": ("scope_calls", "max_scope"),
        "VECTOR_SEARCH": ("vector_calls", "max_vector"),
        "COMMUNITY_SEARCH": ("community_calls", "max_community"),
    }
    if decision.action in caps:
        used_key, cap_key = caps[decision.action]
        if budget[used_key] >= budget[cap_key]:
            alternatives = [
                a
                for a in ("VECTOR_SEARCH", "GRAPH_SEARCH", "GRAPH_SCOPE", "COMMUNITY_SEARCH")
                if a != decision.action
                and budget[caps[a][0]] < budget[caps[a][1]]
            ]
            if alternatives:
                decision.note = (
                    f"{decision.action} budget exhausted; switching to {alternatives[0]}"
                )
                decision.action = alternatives[0]
            else:
                decision.action = "STOP"
                decision.reason_code = "budget_exhausted"
                decision.note = "all tool budgets exhausted"

    return decision, usage or {}
