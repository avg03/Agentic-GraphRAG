"""Query embedding + cosine-similarity retrieval over ChromaDB.

Pipeline:
    query string
      -> embedding via `ingestion.models` (BAAI/bge-small-en-v1.5, 384-dim)
      -> cosine-similarity search against chunks stored in ChromaDB
      -> top-k chunks

ChromaDB collection is created with `{"hnsw:space": "cosine"}` (see
`database/chroma_client.py`), so `collection.query()` already performs
cosine / angular nearest-neighbour search server-side.

Usage:
    from retrieval.retriever import embed_query, retrieve_top_k

    vec = embed_query("What is TigerGraph?")
    hits = retrieve_top_k("What is TigerGraph?", k=5)
"""

from typing import Any, Optional

from ingestion.models import get_embedding_model

__all__ = [
    "embed_query",
    "cosine_similarity",
    "query_collection",
    "retrieve_top_k",
]


# ---------------------------------------------------------------------------
# 1. String -> embedding (uses models.py)
# ---------------------------------------------------------------------------

def embed_query(query: str, model=None) -> list[float]:
    """Convert a query string into an embedding vector.

    Uses the embedding model defined in `ingestion/models.py`
    (BAAI/bge-small-en-v1.5) unless another `model` is passed.

    Supports, in order:
      1. SentenceTransformer-style: `model.encode(...)`
      2. LangChain-style: `model.embed_query(...)` / `model.embed_documents(...)`

    Args:
        query: Raw query string (non-empty).
        model: Optional override; defaults to `get_embedding_model()`.

    Returns:
        Embedding as a list of floats (384-dim for bge-small-en-v1.5).
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    resolved = get_embedding_model(model)

    if hasattr(resolved, "encode"):
        # SentenceTransformer-style (normalized like ingestion path)
        vec = resolved.encode(
            query, normalize_embeddings=True, show_progress_bar=False
        )
        return [float(x) for x in vec]

    if hasattr(resolved, "embed_query"):
        # LangChain-style single-query path (preferred for queries)
        return [float(x) for x in resolved.embed_query(query)]

    if hasattr(resolved, "embed_documents"):
        return [float(x) for x in resolved.embed_documents([query])[0]]

    raise TypeError(
        "Unsupported embedding model interface: expected "
        "`encode(...)` (SentenceTransformer) or "
        "`embed_query(...)`/`embed_documents(...)` (LangChain). "
        f"Got {type(resolved).__name__}."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (1.0 = identical direction)."""
    import math

    if len(a) != len(b):
        raise ValueError(f"Vector dims differ: {len(a)} vs {len(b)}.")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _resolve_collection(collection=None):
    """Lazy import so this module stays importable without a DB on disk."""
    if collection is not None:
        return collection
    from database.chroma_client import get_collection

    return get_collection()


# ---------------------------------------------------------------------------
# 2. Cosine-similarity search over ChromaDB chunks -> top-k
# ---------------------------------------------------------------------------

def query_collection(
    query_embedding: list[float],
    k: int = 5,
    collection=None,
    where: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """Search ChromaDB with a precomputed query embedding (cosine space).

    Args:
        query_embedding: Query vector (must match stored dim, e.g. 384).
        k: Number of top chunks to retrieve.
        collection: Optional ChromaDB collection; defaults to the local
            `tigergraph_docs` collection from `database/chroma_client.py`.
        where: Optional ChromaDB metadata filter, e.g. `{"doc_id": "D1"}`.

    Returns:
        List of dicts, one per hit:
        {"id", "document", "metadata", "distance", "similarity"}
        where `similarity = 1 - distance` (collection uses cosine space).
        Ordered best-first.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not query_embedding:
        raise ValueError("query_embedding must be non-empty.")

    col = _resolve_collection(collection)
    kwargs: dict[str, Any] = {
        "query_embeddings": [list(map(float, query_embedding))],
        "n_results": int(k),
    }
    if where is not None:
        kwargs["where"] = where

    res = col.query(**kwargs)
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: list[dict[str, Any]] = []
    for i, _id in enumerate(ids):
        dist = float(dists[i]) if i < len(dists) else float("nan")
        hits.append(
            {
                "id": _id,
                "document": docs[i] if i < len(docs) else None,
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dist,
                # Chroma cosine distance ~= 1 - cosine_similarity
                "similarity": 1.0 - dist,
            }
        )
    return hits


def retrieve_top_k(
    query: str,
    k: int = 5,
    collection=None,
    model=None,
    where: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """End-to-end: embed `query` string, then retrieve top-k chunks.

    Args:
        query: Raw query string.
        k: Number of top chunks to retrieve (placeholder / tunable).
        collection: Optional ChromaDB collection (default: local DB).
        model: Optional embedding-model override (default: models.py model).
        where: Optional ChromaDB metadata filter.

    Returns:
        Same format as `query_collection` — top-k chunks best-first.
    """
    query_embedding = embed_query(query, model=model)
    return query_collection(
        query_embedding, k=k, collection=collection, where=where
    )


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What is TigerGraph?"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"Query: {q!r} (k={k})")
    for rank, hit in enumerate(retrieve_top_k(q, k=k), start=1):
        doc = (hit["document"] or "")[:200].replace("\n", " ")
        print(
            f"[{rank}] id={hit['id']} "
            f"similarity={hit['similarity']:.4f} "
            f"doc={doc}"
        )
