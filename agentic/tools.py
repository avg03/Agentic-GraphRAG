"""Retrieval tools (v2 §10.2 + lightweight community proxy).

Each tool reuses an existing engine and returns a structured dict; no
retrieval logic lives here.

    graph_search(subquery)      — Pipeline 2 engine, v2 mode dispatch
                                  (entity match -> context resolution ->
                                  local/scope traversal -> ranked chunks)
    scope_search(subquery, …)   — Mode B sibling traversal driven by explicit
                                  context roots (the GRAPH_SCOPE action)
    vector_search(subquery)     — Pipeline 1 engine (BM25 + dense + rerank)
    community_search(subquery)  — lightweight doc-level lookup over Chroma
"""

__all__ = ["graph_search", "scope_search", "vector_search", "community_search"]

_conn = None
_vector_cache: dict = {}


def _get_conn():
    global _conn
    if _conn is None:
        from graphrag.schema import connect

        _conn = connect()
    return _conn


def _tool_base(tool: str) -> dict:
    return {
        "tool": tool,
        "entities": [],
        "paths": [],
        "relationships": [],
        "chunks": [],
        "unmatched": [],
        "retrieval_mode": "",
        "scope_roots": [],
        "question_type": "",
        "status": "success",
    }


def graph_search(subquery: str, max_paths: int = 20) -> dict:
    """Investigate via the existing TigerGraph engine (v2 mode dispatch).

    The v2 orchestrator (graphrag.retrieve.graph_search) handles query
    analysis, entity matching, context resolution, local/scope traversal,
    and scope-aware ranking in one call — this wrapper only adapts the
    result to the tool dict shape.
    """
    from graphrag.retrieve import graph_search as _graph_search

    result = _graph_search(subquery, mode="auto", max_paths=max_paths)
    out = _tool_base("graph_search")
    out.update(
        {
            "entities": [s[2] for s in result["seeds"]],
            "paths": result["relationships"],
            "relationships": result["relationships"],
            "chunks": result["chunks"],
            "unmatched": result["matching"].get("unmatched", []),
            "retrieval_mode": result["retrieval_mode"],
            "scope_roots": result["scope_roots"],
            "question_type": result["question_type"],
            "status": "success" if result["seeds"] or result["scope_roots"] else "no_entity",
        }
    )
    return out


def scope_search(
    subquery: str,
    context_keys: list[str] | None = None,
    max_paths: int = 20,
) -> dict:
    """GRAPH_SCOPE tool (v2 §10.2/§12): investigate siblings of a context root.

    Resolves context roots from the context index (deterministically) and
    traverses the candidate sibling ActionEvents (Mode B). Explicit entity
    seeds from the subquery are still honored and merged.
    """
    from graphrag.context_index import load_context_index, match_contexts
    from graphrag.scope import resolve_scope
    from graphrag.retrieve import graph_search as _graph_search

    keys = list(context_keys or [])
    if not keys:
        # fall back to the subquery text as a scope phrase
        keys = [subquery[:120]]

    index = load_context_index()
    matched = match_contexts(keys, index)
    roots = resolve_scope(matched["matches"])
    resolved_keys = [r["key"] for r in roots]

    result = _graph_search(
        subquery,
        context_keys=resolved_keys,
        mode="scope",
        max_paths=max_paths,
    )
    out = _tool_base("scope_search")
    out.update(
        {
            "entities": [s[2] for s in result["seeds"]],
            "paths": result["relationships"],
            "relationships": result["relationships"],
            "chunks": result["chunks"],
            "unmatched": result["matching"].get("unmatched", []),
            "retrieval_mode": result["retrieval_mode"],
            "scope_roots": result["scope_roots"],
            "question_type": result["question_type"],
            "status": "success" if result["scope_roots"] else "no_scope",
        }
    )
    return out


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


def vector_search(subquery: str, top_k: int = 6, collection=None, filters: dict | None = None) -> dict:
    """Reuse the existing Pipeline 1 stack (BM25 + dense + dedupe + rerank).

    `filters` passes a Chroma `where` filter through (v2 §16:
    vector_search(query, filters=None, top_k=...)), e.g. to constrain a
    fallback to a scope's documents.
    """
    from retrieval.main_pipeline import run_pipeline

    hits = run_pipeline(subquery, collection=collection, where=filters)
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
