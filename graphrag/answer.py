"""Final stage: numbered-context prompt -> Gemini structured GroundedAnswer."""

from shared.llm_client import generate_structured
from shared.metrics import Metrics
from shared.models import GroundedAnswer
from shared.prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt

__all__ = ["generate_answer"]


def generate_answer(question: str, ranked_chunks: list[dict], metrics: Metrics | None = None) -> tuple[GroundedAnswer, dict]:
    """Answer the question from ranked context blocks (validated JSON output)."""
    metrics = metrics or Metrics()
    prompt = build_answer_prompt(question, ranked_chunks)

    answer, usage = generate_structured(
        prompt=prompt,
        response_schema=GroundedAnswer,
        system=ANSWER_SYSTEM_PROMPT,
    )
    from ingestion.chunking import count_tokens

    usage["context_tokens"] = count_tokens(
        "\n\n".join((c.get("document") or "") for c in ranked_chunks)
    )
    latency = usage.pop("_latency_ms", None)
    metrics.record_llm(usage, latency_ms=latency)
    return answer, usage
