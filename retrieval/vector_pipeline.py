"""Pipeline 1 (Vector RAG) — complete: retrieval -> answer with shared metrics.

Reuses the existing retrieval stack (BM25 + dense + dedupe + rerank) and the
SAME answer prompts / LLM client as Pipelines 2 and 3, so the three pipelines
differ ONLY in retrieval — identical model, prompts, and metrics for a fair
comparison.

Usage:
    from retrieval.vector_pipeline import answer_question

    result = answer_question("Who won the gold medal ... ?")
    # same shape as graphrag.retrieval_pipeline.answer_question output
"""

from shared.metrics import Metrics
from shared.models import GroundedAnswer
from shared.prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt

__all__ = ["answer_question"]

DEFAULT_TOP_N = 6


def answer_question(
    query: str,
    top_n: int = DEFAULT_TOP_N,
    collection=None,
    reranker_model=None,
) -> dict:
    """Full Pipeline 1 for one question. Same output shape as Pipeline 2."""
    import time

    from ingestion.chunking import count_tokens
    from retrieval.main_pipeline import run_pipeline
    from shared.llm_client import generate_structured

    metrics = Metrics()

    # ---- retrieval (local models only: BM25 + dense + rerank) ----
    with metrics.timer("retrieval"):
        t0 = time.perf_counter()
        chunks = run_pipeline(
            query,
            n=top_n,
            collection=collection,
            reranker_model=reranker_model,
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000

    for c in chunks:
        c.setdefault("source", "vector")
        c.setdefault("support", 0)

    # ---- answer generation (shared LLM client + shared prompt) ----
    prompt = build_answer_prompt(query, chunks)
    with metrics.timer("llm_answer"):
        t0 = time.perf_counter()
        answer, usage = generate_structured(
            prompt=prompt,
            response_schema=GroundedAnswer,
            system=ANSWER_SYSTEM_PROMPT,
        )
        llm_ms = (time.perf_counter() - t0) * 1000

    usage.pop("_latency_ms", None)
    context_tokens = count_tokens(
        "\n\n".join((c.get("document") or "") for c in chunks)
    )
    metrics.record_llm(usage, latency_ms=llm_ms)
    metrics.incr("final_chunks", len(chunks))
    metrics.incr("citations", len(answer.cited_chunk_ids))

    return {
        "answer": answer,
        "chunks": chunks,
        "metrics": metrics,
        "context_tokens": context_tokens,
        "retrieval_latency_ms": round(retrieval_ms, 1),
        "llm_latency_ms": round(llm_ms, 1),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+")
    args = parser.parse_args()

    for question in args.question:
        print(f"\n=== {question}")
        result = answer_question(question)
        a = result["answer"]
        print(f"answer      : {a.answer}")
        print(f"sufficient  : {a.evidence_sufficient} (confidence {a.confidence})")
        print(f"citations   : {a.cited_chunk_ids}")
        s = result["metrics"].summary()
        llm = s["llm"]
        print(
            f"tokens: context {result['context_tokens']} + in {llm['input_tokens']} / "
            f"out {llm['output_tokens']} = {llm['total_tokens']} total | "
            f"retrieval {result['retrieval_latency_ms']} ms, llm {result['llm_latency_ms']} ms"
        )


if __name__ == "__main__":
    main()
