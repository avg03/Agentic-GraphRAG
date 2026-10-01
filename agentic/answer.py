"""Answer Generator (doc §12): grounded, cited answer over the built context."""

from shared.llm_client import generate_structured
from shared.models import GroundedAnswer
from shared.prompts import ANSWER_SYSTEM_PROMPT

__all__ = ["generate_answer"]


def _build_context_prompt(question: str, chunks: list, graph_paths: list) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        title = c.get("metadata", {}).get("title", "")
        blocks.append(
            f"[{i}] chunk_id={c['id']} (source: {c.get('source', 'graph')}, "
            f"support={c.get('support', 0)}, title: {title})\n{c.get('document', '')}"
        )
    context = "\n\n".join(blocks) or "(no chunks retrieved)"
    path_block = ""
    if graph_paths:
        path_block = (
            "GRAPH EVIDENCE (relationship paths discovered; paths alone are "
            "not proof of facts like winning - rely on chunk text for facts):\n"
            + "\n".join(f"  {p}" for p in graph_paths[:20])
            + "\n\n"
        )
    return (
        f"Question: {question}\n\n"
        f"{path_block}"
        f"Context blocks:\n\n{context}\n\n"
        "Answer as JSON matching the requested schema. Use the exact chunk_id "
        "strings from the block headers in cited_chunk_ids."
    )


def generate_answer(
    question: str,
    chunks: list,
    graph_paths: list,
    metrics=None,
) -> tuple[GroundedAnswer, dict]:
    prompt = _build_context_prompt(question, chunks, graph_paths)
    answer, usage = generate_structured(
        prompt=prompt,
        response_schema=GroundedAnswer,
        system=ANSWER_SYSTEM_PROMPT,
    )
    from ingestion.chunking import count_tokens

    usage["context_tokens"] = count_tokens(
        "\n\n".join((c.get("document") or "") for c in chunks)
    )
    if metrics is not None:
        latency = usage.pop("_latency_ms", None)
        metrics.record_llm(usage, latency_ms=latency)
    elif usage:
        usage.pop("_latency_ms", None)
    return answer, usage or {}
