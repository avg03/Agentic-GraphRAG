"""Supervisor (doc §6+§10 merged): routes the first action and every next action."""

from shared.llm_client import generate_structured
from shared.models import SupervisorDecision
from shared.prompts import SUPERVISOR_SYSTEM_PROMPT, build_supervisor_prompt

__all__ = ["decide_next_action"]

# Hard mapping guard: tools we can still execute under budget.
_VALID_ACTIONS = {"GRAPH_SEARCH", "VECTOR_SEARCH", "COMMUNITY_SEARCH", "REFORMULATE", "STOP"}


def decide_next_action(
    question: str,
    slots: list,
    slot_statuses: dict,
    history: list,
    budget: dict,
    metrics=None,
) -> tuple[SupervisorDecision, dict]:
    """Decide the single next action. Budget is enforced BEFORE the LLM call."""
    # Budget enforcement (§15.10): no LLM call needed to know we must stop.
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
        prompt=build_supervisor_prompt(question, slots, list(slot_statuses.values()), history, budget),
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
        decision.reason_code = "invalid_action"
        decision.note = f"model proposed unknown action; forcing stop"

    # Per-tool budget: fall back to another affordable tool or STOP.
    caps = {
        "GRAPH_SEARCH": ("graph_calls", "max_graph"),
        "VECTOR_SEARCH": ("vector_calls", "max_vector"),
        "COMMUNITY_SEARCH": ("community_calls", "max_community"),
    }
    if decision.action in caps:
        used_key, cap_key = caps[decision.action]
        if budget[used_key] >= budget[cap_key]:
            alternatives = [
                a
                for a in ("VECTOR_SEARCH", "GRAPH_SEARCH", "COMMUNITY_SEARCH")
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
