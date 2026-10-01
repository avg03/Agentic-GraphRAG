"""Deduplicate chunks from dense retrieval + BM25 (O(n)).

Merges the outputs of:
  - `retrieval.retriever.retrieve_top_k` /
    `retrieval.retriever.query_collection`:
      {"id", "document", "metadata", "distance", "similarity"}
  - `retrieval.bm25.bm25_rank` / `retrieval.bm25.rank_chunks_bm25`:
      {"id", "document", "page_content", "metadata", "bm25_score", "rank"}

Goal: chunks appearing in BOTH lists are kept only once (first
occurrence wins, so pass dense hits first to prefer them).

Logic (O(n) time, O(n) space):
    seen_ids = set()
    for chunk in dense_hits + bm25_hits:
        cid = <chunk's id>
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique.append(chunk)

Usage:
    from retrieval.retriever import retrieve_top_k
    from retrieval.bm25 import rank_chunks_bm25
    from retrieval.deduplication import deduplicate_chunks

    dense = retrieve_top_k(query, k=10)
    sparse = rank_chunks_bm25(query, chunks, m=10)
    unique = deduplicate_chunks(dense, sparse)
"""

from typing import Any, Hashable, Iterable

__all__ = [
    "get_chunk_id",
    "deduplicate_chunks",
    "deduplicate",
]


def get_chunk_id(chunk: Any, fallback_index: int = 0) -> Hashable:
    """Extract a stable ID from a chunk dict / Document / string.

    Lookup order for dicts: "id" > metadata.chunk_id > "chunk_id" >
    "page_content"/"document" text. Documents: metadata.chunk_id >
    page_content. Strings: the string itself.
    """
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        if chunk.get("id") is not None:
            return chunk["id"]
        meta = chunk.get("metadata") or {}
        if isinstance(meta, dict) and meta.get("chunk_id") is not None:
            return meta["chunk_id"]
        if chunk.get("chunk_id") is not None:
            return chunk["chunk_id"]
        # Fall back to text identity so identical texts still dedupe.
        for key in ("page_content", "document"):
            if chunk.get(key):
                return chunk[key]
        return f"__index_{fallback_index}"
    # langchain Document (or similar with .metadata / .page_content)
    meta = getattr(chunk, "metadata", None) or {}
    if isinstance(meta, dict) and meta.get("chunk_id") is not None:
        return meta["chunk_id"]
    text = getattr(chunk, "page_content", None)
    if text:
        return text
    return f"__index_{fallback_index}"


def deduplicate_chunks(
    dense_hits: Iterable[Any] | None,
    bm25_hits: Iterable[Any] | None,
) -> list[Any]:
    """Merge dense + BM25 chunk lists, dropping duplicates in O(n).

    Args:
        dense_hits: Chunks from `retriever.py` (top-k), kept first.
        bm25_hits: Chunks from `bm25.py` (top-m), appended if unseen.

    Returns:
        List of unique chunks (first occurrence wins, order preserved).
    """
    seen_ids: set[Hashable] = set()
    unique: list[Any] = []
    index = 0
    for chunk in list(dense_hits or []) + list(bm25_hits or []):
        cid = get_chunk_id(chunk, fallback_index=index)
        index += 1
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique.append(chunk)
    return unique


def deduplicate(*lists: Iterable[Any] | None) -> list[Any]:
    """Generalize to N lists: dedupe preserving first-seen order, O(n)."""
    seen_ids: set[Hashable] = set()
    unique: list[Any] = []
    index = 0
    for lst in lists:
        for chunk in list(lst or []):
            cid = get_chunk_id(chunk, fallback_index=index)
            index += 1
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique.append(chunk)
    return unique


if __name__ == "__main__":
    dense = [
        {"id": "D1_chunk_0", "document": "TigerGraph is a graph DB",
         "metadata": {"chunk_id": "D1_chunk_0"}, "similarity": 0.97},
        {"id": "D2_chunk_0", "document": "Bananas are yellow",
         "metadata": {"chunk_id": "D2_chunk_0"}, "similarity": 0.50},
    ]
    sparse = [
        {"id": "D1_chunk_0", "document": "TigerGraph is a graph DB",
         "metadata": {"chunk_id": "D1_chunk_0"}, "bm25_score": 2.08, "rank": 1},
        {"id": "D3_chunk_0", "document": "GSQL graph queries",
         "metadata": {"chunk_id": "D3_chunk_0"}, "bm25_score": 0.92, "rank": 2},
    ]
    out = deduplicate_chunks(dense, sparse)
    print([c["id"] for c in out])  # ['D1_chunk_0', 'D2_chunk_0', 'D3_chunk_0']
    print(f"{len(dense)} + {len(sparse)} -> {len(out)} unique")
