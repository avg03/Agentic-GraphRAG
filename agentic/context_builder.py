"""Context Builder (doc §11): dedupe evidence, assemble the grounded final context."""

__all__ = ["build_context"]

MAX_CHUNK_CHARS = 1400


def build_context(state: dict, top_n: int = 6) -> tuple[list, list]:
    """Returns (deduped_chunks, graph_paths) for the answer prompt.

    - chunks deduped by id (graph vs vector vs repeated rounds),
    - sorted: graph-supported first, then rerank score,
    - capped to top_n,
    - duplicate retrieval is counted for the trace.
    """
    all_chunks = state.get("chunks", [])
    seen: dict[str, int] = {}
    for c in all_chunks:
        seen[c["id"]] = seen.get(c["id"], 0) + 1
    duplicate_hits = sum(n - 1 for n in seen.values() if n > 1)

    by_id: dict[str, dict] = {}
    for c in all_chunks:
        existing = by_id.get(c["id"])
        if existing is None:
            by_id[c["id"]] = dict(c)
            continue
        # merge: keep best support/source info
        existing["support"] = max(existing.get("support", 0), c.get("support", 0))
        if existing.get("source") != "graph" and c.get("source") == "graph":
            existing["source"] = "graph"

    ordered = sorted(
        by_id.values(),
        key=lambda c: (c.get("support", 0), c.get("rerank_score", 0.0)),
        reverse=True,
    )[:top_n]

    for i, c in enumerate(ordered, 1):
        text = c.get("document") or c.get("page_content") or ""
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS] + " …"
        c["document"] = text

    state["duplicate_chunks_dropped"] = duplicate_hits
    return ordered, state.get("graph_paths", [])
