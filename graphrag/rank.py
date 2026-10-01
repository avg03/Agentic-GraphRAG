"""Rank graph-collected chunks, with a sufficiency gate + dense top-up.

Order of operations:
    1. Support sort — chunks reached by more seeds/events first.
    2. Support gate — if nothing was reached (zero support / no chunks), or
       fewer than MIN_CHUNKS survive, top up with Pipeline-1 dense retrieval
       so the answer stage always has usable context. Every chunk is tagged
       source="graph" | "dense" so eval can report both honestly.
    3. BGE cross-encoder rerank (same model as Pipeline 1 -> fair comparison),
       cutting to top_n.
"""

from graphrag.chunk_lookup import fetch_chunks

__all__ = ["rank_chunks", "DEFAULT_TOP_N", "MIN_CHUNKS"]

DEFAULT_TOP_N = 5
MIN_CHUNKS = 3
DENSE_TOPUP_K = 10  # dense candidates before rerank


def rank_chunks(
    query: str,
    graph_chunk_ids: dict[str, dict],
    top_n: int = DEFAULT_TOP_N,
    collection=None,
    reranker_model=None,
    allow_dense_topup: bool = True,
) -> tuple[list[dict], dict]:
    """Rank graph chunks for the query; returns (ranked_chunks, info).

    Each returned chunk dict carries: id, document, metadata,
    support (int, 0 for dense), source ("graph"|"dense"),
    rerank_score / rerank (from the cross-encoder).
    """
    from retrieval.reranker import rerank

    info: dict = {"graph_chunks": 0, "dense_topup": 0, "gate": "sufficient"}

    graph_chunks: list[dict] = []
    if graph_chunk_ids:
        fetched = fetch_chunks(list(graph_chunk_ids.keys()), collection=collection)
        for c in fetched:
            c["support"] = graph_chunk_ids[c["id"]]["support"]
            c["source"] = "graph"
        graph_chunks = fetched
    info["graph_chunks"] = len(graph_chunks)

    ranked: list[dict] = []
    if graph_chunks:
        best_first = sorted(graph_chunks, key=lambda c: -c["support"])
        reranked = rerank(query, best_first, model=reranker_model, top_n=None)
        ranked = reranked

    # --- sufficiency gate + dense top-up ---
    if allow_dense_topup and len(ranked) < MIN_CHUNKS:
        info["gate"] = "insufficient_graph"
        from retrieval.retriever import retrieve_top_k

        dense_hits = retrieve_top_k(query, k=DENSE_TOPUP_K, collection=collection)
        have = {c["id"] for c in ranked}
        candidates = []
        for hit in dense_hits:
            if hit["id"] in have:
                continue
            candidates.append(
                {
                    "id": hit["id"],
                    "document": hit.get("document", ""),
                    "metadata": hit.get("metadata", {}),
                    "support": 0,
                    "source": "dense",
                }
            )
        if candidates:
            topup = rerank(query, candidates, model=reranker_model, top_n=MIN_CHUNKS - len(ranked))
            info["dense_topup"] = len(topup)
            ranked.extend(topup)
        ranked = sorted(ranked, key=lambda c: c.get("rerank", 10**9))
    elif not ranked:
        info["gate"] = "no_evidence"

    return ranked[:top_n], info
