"""Rank graph-collected chunks (v2 §8/§9): scope-aware scoring + query-aware gate.

Order of operations:
    1. Support sort — chunks reached by more seeds/events first.
    2. BGE cross-encoder rerank (query_relevance; same model as Pipeline 1
       -> fair comparison).
    3. Composite scope-aware score (v2 §8):
           FinalScore = query_relevance
                      + graph_support
                      + context_match      (chunk lineage ∩ requested scope)
       chunks from the same ActionEvent beyond the per-event cap sink below
       distinct-event evidence (duplication penalty) — 20 chunks from one
       event must not outrank 10 events' evidence (v2 §8).
    4. Sufficiency gate + dense top-up (v2 §9): local questions keep the
       MIN_CHUNKS rule; collection questions report scope_coverage
       (partial/substantial/high) and only top up when the graph completely
       missed the scope or too few chunks survive.

Every chunk dict carries: id, document, metadata, support (int, 0 for
dense), source ("graph"|"dense"), event_ids, context_keys,
retrieval_mode, rerank_score / rerank, final_score.
"""

from graphrag import config
from graphrag.normalize import normalize_name
from graphrag.chunk_lookup import fetch_chunks

__all__ = ["rank_chunks", "DEFAULT_TOP_N", "MIN_CHUNKS", "COLLECTION_TOP_N"]

DEFAULT_TOP_N = 5
MIN_CHUNKS = 3
DENSE_TOPUP_K = 10  # dense candidates before rerank
COLLECTION_TOP_N = 16  # collection questions need wide, event-diverse windows

_W_RERANK = 0.60
_W_SUPPORT = 0.25
_W_CONTEXT = 0.15


def _context_match(chunk_keys: list[str], requested_keys: list[str]) -> float:
    """1.0 when the chunk's lineage intersects the requested scope."""
    if not chunk_keys or not requested_keys:
        return 0.0
    requested = {normalize_name(k) for k in requested_keys}
    return 1.0 if any(normalize_name(k) in requested for k in chunk_keys) else 0.0


def _coverage(distinct_events: int, n_chunks: int, question_type: str | None) -> str:
    """scope_coverage verdict (v2 §9): partial / substantial / high."""
    if question_type == "collection_aggregation":
        if distinct_events <= 0:
            return "none"
        if distinct_events >= 10 and n_chunks >= 15:
            return "high"
        if distinct_events >= 5 and n_chunks >= 8:
            return "substantial"
        return "partial"
    return "high" if n_chunks >= MIN_CHUNKS else "partial"


def rank_chunks(
    query: str,
    graph_chunk_ids: dict[str, dict],
    top_n: int = DEFAULT_TOP_N,
    collection=None,
    reranker_model=None,
    allow_dense_topup: bool = True,
    question_type: str | None = None,
    requested_keys: list[str] | None = None,
    retrieval_mode: str = "local",
) -> tuple[list[dict], dict]:
    """Rank graph chunks for the query; returns (ranked_chunks, info).

    graph_chunk_ids: {chunk_id: {support, via, event_ids, context_keys?}}
    question_type: from the query analysis — None behaves as a local
        question (v1 gate semantics).
    requested_keys: context keys the query resolved to (context_match).
    """
    from retrieval.reranker import rerank

    is_collection = question_type == "collection_aggregation"
    requested_keys = requested_keys or []
    max_per_event = (
        config.SCOPE_MAX_CHUNKS_PER_EVENT if is_collection else MIN_CHUNKS
    )

    info: dict = {
        "graph_chunks": 0,
        "dense_topup": 0,
        "gate": "sufficient",
        "scope_coverage": "none",
        "distinct_events": 0,
        "duplicate_chunk_count": 0,
        "retrieval_mode": retrieval_mode,
    }

    graph_chunks: list[dict] = []
    if graph_chunk_ids:
        fetched = fetch_chunks(list(graph_chunk_ids.keys()), collection=collection)
        for c in fetched:
            entry = graph_chunk_ids[c["id"]]
            c["support"] = entry.get("support", 0)
            c["source"] = "graph"
            c["event_ids"] = list(entry.get("event_ids", []))
            c["context_keys"] = list(entry.get("context_keys", []))
            c["retrieval_mode"] = retrieval_mode
        graph_chunks = fetched
    info["graph_chunks"] = len(graph_chunks)

    ranked: list[dict] = []
    if graph_chunks:
        best_first = sorted(graph_chunks, key=lambda c: -c["support"])
        reranked = rerank(query, best_first, model=reranker_model, top_n=None)

        # composite scope-aware score (v2 §8)
        for c in reranked:
            c["context_match"] = _context_match(c.get("context_keys", []), requested_keys)
            c["final_score"] = round(
                _W_RERANK * float(c.get("rerank_score", 0.0))
                + _W_SUPPORT * min(c.get("support", 0), 5) / 5.0
                + _W_CONTEXT * c["context_match"],
                4,
            )
        reranked.sort(key=lambda c: -c["final_score"])

        # duplication penalty: beyond the per-event cap, same-event chunks
        # sink below the first chunks of not-yet-covered events
        per_event_count: dict[str, int] = {}
        for c in reranked:
            c["_event_tag"] = c["event_ids"][0] if c.get("event_ids") else ""
            if c["_event_tag"]:
                seen = per_event_count.get(c["_event_tag"], 0)
                per_event_count[c["_event_tag"]] = seen + 1
                c["_sunk"] = seen >= max_per_event
            else:
                c["_sunk"] = False
        ranked = reranked
        info["distinct_events"] = len(per_event_count)
        info["duplicate_chunk_count"] = sum(1 for c in ranked if c.get("_sunk"))
    info["scope_coverage"] = _coverage(info["distinct_events"], len(ranked), question_type)

    # --- sufficiency gate + dense top-up (v2 §9/§15.7) ---
    # Collection questions: dense top-up only on a (near-)complete graph
    # miss — sibling evidence must come from the scope traversal.
    # Local questions: the graph finds the right branch, but the exact fact
    # often lives in an infobox/results chunk the graph ranked low or missed
    # (v2 §15.7: "the chunk remains the primary grounding source for exact
    # tabular values") — fill the remaining context slots with dense hits.
    if allow_dense_topup:
        if is_collection:
            want = MIN_CHUNKS if len(ranked) < MIN_CHUNKS else 0
        else:
            want = max(0, top_n - len(ranked))
        if want > 0:
            info["gate"] = "insufficient_graph" if is_collection else "dense_supplemented"
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
                        "event_ids": [],
                        "context_keys": [],
                        "retrieval_mode": retrieval_mode,
                    }
                )
            if candidates:
                topup = rerank(query, candidates, model=reranker_model, top_n=want)
                info["dense_topup"] = len(topup)
                ranked.extend(topup)
        elif not ranked:
            info["gate"] = "no_evidence"
    elif not ranked:
        info["gate"] = "no_evidence"

    # final order: rerank position within each tier (sunk duplicates demoted)
    ranked.sort(key=lambda c: (1 if c.get("_sunk") else 0, c.get("rerank", 10**9)))
    for c in ranked:
        c.pop("_sunk", None)
        c.pop("_event_tag", None)

    return ranked[:top_n], info
