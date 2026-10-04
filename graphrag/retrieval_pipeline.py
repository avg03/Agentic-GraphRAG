"""Pipeline 2 driver: question -> graph-grounded, cited, structured answer.

Stages (v2 architecture; each timed + counted via shared.metrics):
    1. Query entity + scope analysis   Gemini structured (1 LLM call)
    2. Entity matching                 normalized + BGE similarity -> seeds
    3. Context resolution              candidate keys -> ranked scope roots
    4. Graph traversal                 local (Mode A) and/or scope/sibling
                                       (Mode B, collection questions)
    5. Rank + sufficiency gate         scope-aware composite + dense top-up
    6. Answer generation               Gemini structured GroundedAnswer (1 LLM call)

Usage:
    from graphrag.retrieval_pipeline import answer_question

    result = answer_question("Who won the gold medal in ...?")
    print(result["answer"].answer, result["answer"].cited_chunk_ids)

    python -m graphrag.retrieval_pipeline "question here"
"""

from shared.metrics import Metrics

from graphrag.answer import generate_answer
from graphrag.rank import DEFAULT_TOP_N, MIN_CHUNKS
from graphrag.retrieve import graph_search

__all__ = ["answer_question"]

_RETRIEVAL_STAGES = (
    "stage2_entity_matching",
    "stage3_context_resolution",
    "stage4_traverse",
    "stage4b_relationships",
    "stage5_rank",
)
_LLM_STAGES = ("stage1_query_analysis",)


def answer_question(
    query: str,
    top_n: int = DEFAULT_TOP_N,
    min_chunks: int = MIN_CHUNKS,  # kept for CLI compat; gate lives in rank
    allow_dense_topup: bool = True,
    collection=None,
    conn=None,
    mode: str = "auto",
) -> dict:
    """Full Pipeline 2 for one question. Returns answer + chunks + metrics.

    The return dict keeps the v1 keys (answer/chunks/seeds/matching/
    rank_info/metrics/context_tokens/retrieval_latency_ms/llm_latency_ms)
    and adds v2 fields (question_type, retrieval_mode, scope_roots,
    relationships).
    """
    metrics = Metrics()

    # 1-5. analysis -> matching -> context resolution -> traversal -> rank
    retrieval = graph_search(
        query,
        mode=mode,
        top_k=top_n,
        collection=collection,
        conn=conn,
        metrics=metrics,
        allow_dense_topup=allow_dense_topup,
    )
    ranked = retrieval["chunks"]
    rank_info = retrieval["rank_info"]

    # 6. Grounded answer (Gemini)
    with metrics.timer("stage6_answer"):
        answer, answer_usage = generate_answer(query, ranked, metrics)
    metrics.incr("citations", len(answer.cited_chunk_ids))

    timings = metrics.summary().get("timings", {})
    return {
        "answer": answer,
        "chunks": ranked,
        "seeds": retrieval["seeds"],
        "matching": retrieval["matching"],
        "rank_info": rank_info,
        "metrics": metrics,
        "context_tokens": answer_usage.get("context_tokens", 0),
        # v2 fields (additive)
        "question_type": retrieval["question_type"],
        "retrieval_mode": retrieval["retrieval_mode"],
        "scope_roots": retrieval["scope_roots"],
        "relationships": retrieval["relationships"],
        # query-time retrieval = matching + context + traversal + ranking
        "retrieval_latency_ms": round(sum(
            timings.get(n, {}).get("total_ms", 0)
            for n in _RETRIEVAL_STAGES
        ), 1),
        # LLM latency = query analysis + answer generation
        "llm_latency_ms": round(sum(
            timings.get(n, {}).get("total_ms", 0) for n in _LLM_STAGES
        ) + timings.get("stage6_answer", {}).get("total_ms", 0), 1),
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="Question(s) to answer")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--mode", default="auto", choices=["auto", "local", "scope"])
    args = parser.parse_args()

    for question in args.question:
        print(f"\n=== {question}")
        result = answer_question(question, top_n=args.top_n, mode=args.mode)
        a = result["answer"]
        print(f"answer     : {a.answer}")
        print(f"sufficient : {a.evidence_sufficient} (confidence {a.confidence:.2f})")
        print(f"citations  : {a.cited_chunk_ids}")
        print(f"seeds      : {[(t, i) for t, i, _ in result['seeds']]}")
        print(
            f"mode       : {result['retrieval_mode']} ({result['question_type']})"
            f" coverage={result['rank_info'].get('scope_coverage')}"
        )
        print(f"rank gate  : {result['rank_info']}")
        print("metrics    :")
        print(result["metrics"].pretty())


if __name__ == "__main__":
    main()
