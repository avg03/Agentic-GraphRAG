"""Fetch chunk text from ChromaDB by chunk ID (graph -> chunk text bridge)."""

from database.chroma_client import get_collection

__all__ = ["fetch_chunks"]


def fetch_chunks(chunk_ids: list[str], collection=None) -> list[dict]:
    """Fetch chunks by ID from Chroma.

    Returns a list of {"id", "document", "metadata"} in Chroma's order;
    missing IDs are reported (they exist in the graph but not the vector DB).
    """
    if not chunk_ids:
        return []
    if collection is None:
        collection = get_collection()

    stored = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
    ids = stored.get("ids") or []
    docs = stored.get("documents") or []
    metas = stored.get("metadatas") or []

    chunks = []
    for i, cid in enumerate(ids):
        chunks.append(
            {
                "id": cid,
                "document": docs[i] if i < len(docs) else "",
                "metadata": dict(metas[i]) if i < len(metas) and metas[i] else {},
            }
        )

    missing = set(chunk_ids) - set(ids)
    if missing:
        print(f"WARNING: {len(missing)} chunk IDs in graph not found in Chroma: {sorted(missing)[:5]}")
    return chunks
