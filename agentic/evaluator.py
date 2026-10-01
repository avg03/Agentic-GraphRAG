"""Evidence Evaluator (doc §9): per-slot support matrix, contradictions, gaps."""

from shared.llm_client import generate_structured
from shared.models import EvidenceEvaluation
from shared.prompts import EVALUATOR_SYSTEM_PROMPT, build_evaluator_prompt

__all__ = ["evaluate_evidence"]


def evaluate_evidence(
    question: str,
    slots: list,
    graph_paths: list,
    chunks: list,
    unmatched: list,
    metrics=None,
) -> tuple[EvidenceEvaluation, dict]:
    """Judge each answer slot against the collected evidence (1 Gemini call)."""
    evaluation, usage = generate_structured(
        prompt=build_evaluator_prompt(question, slots, graph_paths, chunks, unmatched),
        response_schema=EvidenceEvaluation,
        system=EVALUATOR_SYSTEM_PROMPT,
    )
    if metrics is not None:
        latency = usage.pop("_latency_ms", None)
        metrics.record_llm(usage, latency_ms=latency)
    elif usage:
        usage.pop("_latency_ms", None)

    # Trustworthy defaults: evaluator may skip slots.
    seen = {s.slot_id for s in evaluation.slots}
    for slot in slots:
        if slot["id"] not in seen:
            from shared.models import SlotStatus

            evaluation.slots.append(
                SlotStatus(slot_id=slot["id"], status="missing", support_score=0.0)
            )
    return evaluation, usage or {}
