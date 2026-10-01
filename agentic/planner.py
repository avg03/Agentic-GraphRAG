"""Node 1 — Planner: query understanding + decomposition into answer slots."""

from shared.llm_client import generate_structured
from shared.models import PlanOutput
from shared.prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt

__all__ = ["plan_question"]


def plan_question(question: str, metrics=None) -> tuple[PlanOutput, dict]:
    """Understand the question; returns (PlanOutput, usage_dict)."""
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
