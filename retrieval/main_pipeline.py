"""End-to-end retrieval pipeline: query string -> final top-N chunks.

Steps (each reuses an existing module, no duplicated logic):
    1. Clean          (`retrieval.query.clean_query`)
    2. Dense top-K    (`retrieval.retriever.retrieve_top_k` over ChromaDB)
    3. Sparse top-M   (`retrieval.bm25.rank_chunks_bm25` over a chunk corpus)
    4. Deduplicate    (`retrieval.deduplication.deduplicate_chunks`)
    5. Rerank top-N   (`retrieval.reranker.rerank` with BGE CrossEncoder)

Usage:
    from retrieval.main_pipeline import run_pipeline

    final = run_pipeline("What is TigerGraph?")
    # -> list of top-N reranked chunk dicts (best-first)
"""

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Tunable placeholders — fill these in.
# ---------------------------------------------------------------------------

TOP_K = 20  # dense retrieval: retriever top-k chunks
TOP_M = 20  # sparse retrieval: BM25 top-m chunks
TOP_N = 5   # final rerank: cross-encoder top-n chunks

__all__ = [
    "TOP_K",
    "TOP_M",
    "TOP_N",
    "load_bm25_corpus_from_chroma",
    "run_pipeline",
    "main",
]


# ---------------------------------------------------------------------------
# BM25 corpus fallback: full chunk list pulled from ChromaDB.
# ---------------------------------------------------------------------------

def load_bm25_corpus_from_chroma(collection=None) -> list[dict[str, Any]]:
    """Load all stored chunks from Chroma as BM25-ready dicts.

    Args:
        collection: Optional ChromaDB collection. Defaults to the local
            ``tigergraph_docs`` collection from ``database/chroma_client.py``.

    Returns:
        List of ``{"id", "document", "page_content", "metadata"}`` dicts,
        accepted directly by ``bm25.rank_chunks_bm25``.
    """
    if collection is None:
        from database.chroma_client import get_collection

        collection = get_collection()

    stored = collection.get(include=["documents", "metadatas"])
    ids = stored.get("ids") or []
    docs = stored.get("documents") or []
    metas = stored.get("metadatas") or []

    corpus: list[dict[str, Any]] = []
    for i, _id in enumerate(ids):
        text = docs[i] if i < len(docs) else ""
        meta = dict(metas[i]) if i < len(metas) and metas[i] else {}
        corpus.append(
            {
                "id": _id,
                "document": text,
                "page_content": text,
                "metadata": meta,
            }
        )
    return corpus


# ---------------------------------------------------------------------------
# Main: query string -> final top-N chunks
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    k: int = TOP_K,
    m: int = TOP_M,
    n: int = TOP_N,
    bm25_corpus=None,
    collection=None,
    embedding_model=None,
    reranker_model=None,
    where: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """Run the full pipeline for one query string.

    Args:
        query: Raw user query (cleaned internally via ``clean_query``).
        k: Dense count — top-k chunks from Chroma (default ``TOP_K``).
        m: Sparse count — top-m chunks from BM25 (default ``TOP_M``).
        n: Final count — top-n chunks after cross-encoder rerank
            (default ``TOP_N``). This is the length of the returned list.
        bm25_corpus: Chunk list for BM25 (Documents / dicts / strings).
            If None, loaded from Chroma via ``load_bm25_corpus_from_chroma``.
        collection: Optional ChromaDB collection for dense retrieval
            (default: local ``tigergraph_docs`` collection).
        embedding_model: Optional embedding-model override (default:
            ``ingestion.models`` BAAI/bge-small-en-v1.5).
        reranker_model: Optional CrossEncoder override (default:
            ``ingestion.models`` BAAI/bge-reranker-base).
        where: Optional ChromaDB metadata filter for the dense step,
            e.g. ``{"doc_id": "D1"}``.

    Returns:
        Final top-n reranked chunk dicts, best-first. Each dict carries
        its dense (``similarity``/``distance``) and/or sparse
        (``bm25_score``) fields plus rerank fields
        (``rerank_score``, ``rerank_logit``, ``rerank``).

    Raises:
        TypeError: If ``query`` is not a string.
        ValueError: If query is empty (even after cleaning), or if
            k / m / n are not positive integers.
    """
    from retrieval.bm25 import rank_chunks_bm25
    from retrieval.deduplication import deduplicate_chunks
    from retrieval.query import clean_query
    from retrieval.reranker import rerank
    from retrieval.retriever import retrieve_top_k

    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}.")
    for name, val in (("k", k), ("m", m), ("n", n)):
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"{name} must be a positive integer, got {val!r}.")

    # 1. Clean
    cleaned = clean_query(query)
    if not cleaned:
        raise ValueError("query must be a non-empty string (after cleaning).")

    # 2. Dense top-k (embeds the cleaned query internally)
    dense_hits = retrieve_top_k(
        cleaned, k=k, collection=collection,
        model=embedding_model, where=where,
    )

    # 3. Sparse top-m (explicit corpus, or full Chroma dump as fallback)
    corpus = bm25_corpus
    if corpus is None:
        if collection is None:
            from database.chroma_client import get_collection

            collection = get_collection()
        corpus = load_bm25_corpus_from_chroma(collection)
    bm25_hits = rank_chunks_bm25(cleaned, corpus, m=m)

    # 4. Deduplicate (dense first, so dense fields win on overlap)
    unique = deduplicate_chunks(dense_hits, bm25_hits)
    if not unique:
        return []

    # 5. Cross-encoder rerank -> final top-n
    return rerank(cleaned, unique, model=reranker_model, top_n=n)


# Backwards-friendly alias: single obvious entry point name.
def main(query: str, **kwargs) -> list[dict[str, Any]]:
    """Alias of :func:`run_pipeline`."""
    return run_pipeline(query, **kwargs)


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What is TigerGraph?"
    print(f"Query: {q!r} (k={TOP_K}, m={TOP_M}, n={TOP_N})")
    for hit in run_pipeline(q):
        text = (hit.get("document") or hit.get("page_content") or "")[:160]
        text = text.replace("\n", " ")
        print(
            f"[rerank {hit.get('rerank')}] id={hit.get('id')} "
            f"score={hit.get('rerank_score', float('nan')):.4f} "
            f"text={text}"
        )
