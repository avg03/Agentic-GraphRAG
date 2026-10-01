"""Cross-encoder reranking over deduplicated chunk hit-dicts (BGE reranker).

Takes the merged unique chunks from `retrieval/deduplication.py`
(which combine `retriever.py` dense hits and `bm25.py` sparse hits):

    {"id", "document", "metadata", ...}

scores each (query, chunk-text) pair with the reranker model from
`ingestion/models.py` (BAAI/bge-reranker-base CrossEncoder), and
returns the re-ranked chunks best-first.

Usage:
    from retrieval.deduplication import deduplicate_chunks
    from retrieval.reranker import rerank

    unique = deduplicate_chunks(dense_hits, bm25_hits)
    ranked = rerank("What is TigerGraph?", unique)
    # [{..., "rerank_score", "rerank"}, ...] best-first
"""

from typing import Any, Optional

from ingestion.models import get_reranker_model

__all__ = [
    "get_chunk_text",
    "rerank_scores",
    "rerank",
]


def get_chunk_text(chunk: dict[str, Any]) -> str:
    """Extract rerankable text from a hit-dict (document > page_content)."""
    if not isinstance(chunk, dict):
        raise TypeError(
            "rerank() expects hit-dicts from deduplication.py, "
            f"got {type(chunk).__name__}."
        )
    text = chunk.get("document") or chunk.get("page_content") or ""
    return text


def rerank_scores(
    query: str,
    chunks: list[dict[str, Any]],
    model=None,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> list[float]:
    """Score (query, chunk) pairs with the CrossEncoder; raw logits.

    Args:
        query: Raw query string (non-empty).
        chunks: Hit-dicts from `deduplication.py` (non-empty texts).
        model: Optional CrossEncoder override (default: models.py reranker).
        batch_size: CrossEncoder.predict batch size.
        show_progress_bar: Show scoring progress bar.

    Returns:
        Raw relevance logits, one per chunk, in input order.
    """
    import math

    _ = math  # keep import local intent clear (sigmoid applied in rerank)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not chunks:
        return []
    texts = [get_chunk_text(c) for c in chunks]
    if any(not t.strip() for t in texts):
        raise ValueError("Cannot rerank chunk with empty document text.")
    resolved = get_reranker_model(model)
    pairs = [(query, text) for text in texts]
    scores = resolved.predict(
        pairs, batch_size=batch_size, show_progress_bar=show_progress_bar
    )
    return [float(s) for s in scores]


def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    model=None,
    top_n: Optional[int] = None,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> list[dict[str, Any]]:
    """Rerank deduped chunk hit-dicts with the BGE CrossEncoder.

    Args:
        query: Raw query string.
        chunks: Hit-dicts from `deduplication.py` /
            `deduplicate_chunks(dense_hits, bm25_hits)`.
        model: Optional CrossEncoder override (default: models.py reranker).
        top_n: Optional cut — return only the top-n reranked chunks
            (default None = return all, reordered best-first).
        batch_size: CrossEncoder.predict batch size.
        show_progress_bar: Show scoring progress bar.

    Returns:
        New list (input dicts copied, best-first), each enriched with:
        {"rerank_score" (sigmoid 0-1), "rerank_logit" (raw), "rerank" (1-based)}.
    """
    import math

    if not chunks:
        return []
    if top_n is not None and (not isinstance(top_n, int) or top_n <= 0):
        raise ValueError("top_n must be a positive integer or None.")

    logits = rerank_scores(
        query, chunks, model=model,
        batch_size=batch_size, show_progress_bar=show_progress_bar,
    )
    rescored: list[dict[str, Any]] = []
    for chunk, logit in zip(chunks, logits):
        out = dict(chunk)  # copy — never mutate caller's dicts
        out["rerank_logit"] = float(logit)
        out["rerank_score"] = 1.0 / (1.0 + math.exp(-float(logit)))
        rescored.append(out)

    rescored.sort(key=lambda h: h["rerank_score"], reverse=True)
    if top_n is not None:
        rescored = rescored[: min(top_n, len(rescored))]
    for rank, hit in enumerate(rescored, start=1):
        hit["rerank"] = rank
    return rescored


if __name__ == "__main__":
    unique = [
        {"id": "D1_chunk_0", "document": "TigerGraph is a graph database.",
         "metadata": {"chunk_id": "D1_chunk_0"}, "similarity": 0.97},
        {"id": "D2_chunk_0", "document": "Bananas are yellow fruit.",
         "metadata": {"chunk_id": "D2_chunk_0"}, "similarity": 0.50},
        {"id": "D3_chunk_0", "document": "TigerGraph supports GSQL graph queries.",
         "metadata": {"chunk_id": "D3_chunk_0"}, "bm25_score": 0.92},
    ]
    for hit in rerank("What is TigerGraph?", unique):
        print(
            f"[rerank {hit['rerank']}] {hit['id']} "
            f"score={hit['rerank_score']:.4f}"
        )
