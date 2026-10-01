"""Pipeline 2 driver: question -> graph-grounded, cited, structured answer.

Stages (each timed + counted via shared.metrics):
    1. Query entity extraction   Gemini structured (1 LLM call)
    2. Entity matching           normalized + BGE similarity -> seed vertices
    3. Graph traversal           TigerGraph -> chunks with support scores
    4. Chunk lookup              Chroma get-by-ID
    5. Rank + sufficiency gate   graph support -> BGE rerank -> dense top-up
    6. Answer generation         Gemini structured GroundedAnswer (1 LLM call)

Usage:
    from graphrag.retrieval_pipeline import answer_question

    result = answer_question("Who won the gold medal in ...?")
    print(result["answer"].answer, result["answer"].cited_chunk_ids)

    python -m graphrag.retrieval_pipeline "question here"
"""

from shared.metrics import Metrics
from shared.prompts import QUERY_ENTITY_SYSTEM_PROMPT, build_query_entity_prompt
from shared.models import QueryEntities

from graphrag.answer import generate_answer
from graphrag.entity_index import load_entity_index, match_entities
from graphrag.rank import DEFAULT_TOP_N, MIN_CHUNKS, rank_chunks
from graphrag.traverse import traverse

__all__ = ["answer_question"]

_entity_index = None


def _get_index(conn=None):
    global _entity_index
    if _entity_index is None:
        if conn is None:
            from graphrag.schema import connect

            conn = connect()
        _entity_index = load_entity_index(conn)
    return _entity_index


def answer_question(
    query: str,
    top_n: int = DEFAULT_TOP_N,
    min_chunks: int = MIN_CHUNKS,
    allow_dense_topup: bool = True,
    collection=None,
    conn=None,
) -> dict:
    """Full Pipeline 2 for one question. Returns answer + chunks + metrics."""
    metrics = Metrics()

    # 1. Query entity extraction (Gemini)
    with metrics.timer("stage1_query_entities"):
        parsed, usage = None, None
        from shared.llm_client import generate_structured

        parsed, usage = generate_structured(
            prompt=build_query_entity_prompt(query),
            response_schema=QueryEntities,
            system=QUERY_ENTITY_SYSTEM_PROMPT,
        )
        usage.pop("_latency_ms", None)
        metrics.record_llm(usage)
    mentions = [e.model_dump() for e in parsed.entities]
    metrics.incr("query_entities", len(mentions))

    # 2. Entity matching -> seeds
    with metrics.timer("stage2_entity_matching"):
        index = _get_index(conn)
        matching = match_entities(mentions, index)
    seeds = matching["seeds"]
    metrics.incr("seed_entities", len(seeds))
    metrics.incr("unmatched_mentions", len(matching["unmatched"]))

    # 3. Graph traversal
    graph_chunk_ids: dict = {}
    if seeds:
        if conn is None:
            from graphrag.schema import connect

            conn = connect()
        with metrics.timer("stage3_traverse"):
            graph_chunk_ids = traverse(conn, seeds)
    metrics.incr("graph_chunks", len(graph_chunk_ids))

    # 4+5. Chunk lookup, support sort, rerank, gate, top-up
    with metrics.timer("stage45_rank"):
        ranked, rank_info = rank_chunks(
            query,
            graph_chunk_ids,
            top_n=top_n,
            collection=collection,
            allow_dense_topup=allow_dense_topup,
        )
    metrics.incr("dense_topup_chunks", rank_info["dense_topup"])
    metrics.counters["rank_gate"] = rank_info["gate"]
    metrics.incr("final_chunks", len(ranked))

    # 6. Grounded answer (Gemini)
    with metrics.timer("stage6_answer"):
        answer, answer_usage = generate_answer(query, ranked, metrics)
    metrics.incr("citations", len(answer.cited_chunk_ids))

    timings = metrics.summary().get("timings", {})
    return {
        "answer": answer,
        "chunks": ranked,
        "seeds": seeds,
        "matching": matching,
        "rank_info": rank_info,
        "metrics": metrics,
        "context_tokens": answer_usage.get("context_tokens", 0),
        # query-time retrieval = matching + traversal + ranking
        "retrieval_latency_ms": round(sum(
            timings.get(n, {}).get("total_ms", 0)
            for n in ("stage2_entity_matching", "stage3_traverse", "stage45_rank")
        ), 1),
        # LLM latency = query-entity extraction + answer generation
        "llm_latency_ms": round(
            timings.get("stage1_query_entities", {}).get("total_ms", 0)
            + timings.get("stage6_answer", {}).get("total_ms", 0), 1),
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="Question(s) to answer")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()

    for question in args.question:
        print(f"\n=== {question}")
        result = answer_question(question, top_n=args.top_n)
        a = result["answer"]
        print(f"answer     : {a.answer}")
        print(f"sufficient : {a.evidence_sufficient} (confidence {a.confidence:.2f})")
        print(f"citations  : {a.cited_chunk_ids}")
        print(f"seeds      : {[(t, i) for t, i, _ in result['seeds']]}")
        print(f"rank gate  : {result['rank_info']}")
        print("metrics    :")
        print(result["metrics"].pretty())


if __name__ == "__main__":
    main()
