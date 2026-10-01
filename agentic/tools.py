"""Retrieval tools (doc §7/§8 + lightweight §6 community proxy).

Each tool reuses an existing engine and returns a structured dict; no
retrieval logic lives here.

    graph_search(subquery)      — Pipeline 2 engine (entity match → traverse →
                                  relationship paths → chunk lookup)
    vector_search(subquery)     — Pipeline 1 engine (BM25 + dense + rerank)
    community_search(subquery)  — lightweight doc-level lookup over Chroma
"""

__all__ = ["graph_search", "vector_search", "community_search"]

_conn = None
_vector_cache: dict = {}


def _get_conn():
    global _conn
    if _conn is None:
        from graphrag.schema import connect

        _conn = connect()
    return _conn


def graph_search(subquery: str, max_paths: int = 20) -> dict:
    """Investigate relationships via the existing TigerGraph engine."""
    from graphrag.entity_index import load_entity_index, match_entities
    from graphrag.traverse import traverse, collect_relationships
    from graphrag.chunk_lookup import fetch_chunks

    conn = _get_conn()
    index = load_entity_index(conn)

    extraction = _extract_query_entities(subquery)
    mentions = [e.model_dump() for e in extraction.entities]
    # The subquery itself is also a match candidate: canonical event names
    # live in vertex names, and planner mentions are sometimes too coarse
    # ("Canoeing") to reach the specific event vertex ("Canoeing at the ...
    # - Men's K-2 1000 metres").
    mentions.append({"name": subquery[:220], "type": None})
    matching = match_entities(mentions, index)
    seeds = matching["seeds"]

    result = {
        "tool": "graph_search",
        "entities": [s[2] for s in seeds],
        "paths": [],
        "relationships": [],
        "chunks": [],
        "unmatched": matching["unmatched"],
        "status": "success" if seeds else "no_entity",
    }
    if not seeds:
        return result

    supports = traverse(conn, seeds)
    relationships = collect_relationships(conn, seeds, max_paths=max_paths)
    chunks = fetch_chunks(list(supports.keys()))
    for c in chunks:
        c["support"] = supports[c["id"]]["support"]
        c["source"] = "graph"
    chunks.sort(key=lambda c: -c["support"])

    result["paths"] = relationships
    result["relationships"] = relationships
    result["chunks"] = chunks
    return result


def _extract_query_entities(subquery: str):
    """One Gemini structured call: named entities in the subquery."""
    from shared.llm_client import generate_structured
    from shared.models import QueryEntities
    from shared.prompts import QUERY_ENTITY_SYSTEM_PROMPT, build_query_entity_prompt

    parsed, _usage = generate_structured(
        prompt=build_query_entity_prompt(subquery),
        response_schema=QueryEntities,
        system=QUERY_ENTITY_SYSTEM_PROMPT,
    )
    return parsed


def vector_search(subquery: str, top_k: int = 6, collection=None) -> dict:
    """Reuse the existing Pipeline 1 stack (BM25 + dense + dedupe + rerank)."""
    from retrieval.main_pipeline import run_pipeline

    hits = run_pipeline(subquery, collection=collection)
    for h in hits:
        h["source"] = "vector"
        h["support"] = 0
    return {
        "tool": "vector_search",
        "chunks": hits,
        "scores": [h.get("rerank_score") for h in hits],
        "citations": [h.get("id") for h in hits],
        "status": "success" if hits else "empty",
    }


def community_search(subquery: str, top_docs: int = 5, collection=None) -> dict:
    """Lightweight community proxy: BM25 over chunks, grouped by document.

    Returns top "communities" = document clusters with title/url/sample text.
    Deliberately shallow: broad-lookup option, not real community detection.
    """
    from collections import defaultdict
    from retrieval.bm25 import rank_chunks_bm25
    from retrieval.main_pipeline import load_bm25_corpus_from_chroma
    from retrieval.query import clean_query

    if collection is None:
        from database.chroma_client import get_collection

        collection = get_collection()

    corpus = load_bm25_corpus_from_chroma(collection)
    cleaned = clean_query(subquery) or subquery
    hits = rank_chunks_bm25(cleaned, corpus, m=top_docs * 4)

    docs: dict[str, dict] = defaultdict(
        lambda: {"title": "", "url": "", "chunk_count": 0, "best_rank": 10**9, "sample": ""}
    )
    for h in hits:
        meta = h.get("metadata", {})
        doc_id = meta.get("doc_id", "")
        if not doc_id:
            continue
        d = docs[doc_id]
        d["title"] = meta.get("title", "")
        d["url"] = meta.get("url", meta.get("source", ""))
        d["chunk_count"] += 1
        d["best_rank"] = min(d["best_rank"], h.get("rank", 10**9))
        if not d["sample"]:
            d["sample"] = (h.get("document") or "")[:400]

    ranked = sorted(docs.items(), key=lambda kv: kv[1]["best_rank"])[:top_docs]
    return {
        "tool": "community_search",
        "communities": [
            {"doc_id": doc_id, **info} for doc_id, info in ranked
        ],
        "status": "success" if ranked else "empty",
    }
