"""Node 1 — Planner (v2 §10.2): query understanding + slots + scope hypotheses."""

from shared.llm_client import generate_structured
from shared.models import PlanOutput
from shared.prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt

__all__ = ["plan_question"]


def plan_question(question: str, metrics=None) -> tuple[PlanOutput, dict]:
    """Understand the question; returns (PlanOutput, usage_dict).

    One LLM call produces BOTH the answer-slot decomposition and the scope
    analysis (question_type / scope_root / candidate_context_keys, v2 §10.2
    step 1) — the context resolver node then resolves the keys offline.
    """
    plan, usage = generate_structured(
        prompt=build_planner_prompt(question),
        response_schema=PlanOutput,
        system=PLANNER_SYSTEM_PROMPT,
    )
    if metrics is not None:
        latency = usage.pop("_latency_ms", None)
        metrics.record_llm(usage, latency_ms=latency)
    elif usage:
        usage.pop("_latency_ms", None)
    if not plan.answer_slots:
        # Never return an empty plan.
        from shared.models import AnswerSlot

        plan.answer_slots = [
            AnswerSlot(id="s1", question=question, slot_type="relational")
        ]
        plan.complexity = "low"
    return plan, usage or {}
