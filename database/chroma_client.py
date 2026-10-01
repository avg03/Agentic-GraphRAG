"""Local persistent ChromaDB setup.

Creates a ChromaDB database locally on disk (via PersistentClient)
and establishes a connection to a named collection you can reuse
across ingestion / retrieval:

    from database.chroma_client import get_collection, get_client

    collection = get_collection()  # defaults to "tigergraph_docs"
    collection.add(ids=[...], documents=[...], embeddings=[...], metadatas=[...])

Persist directory defaults to `<project_root>/chroma_db`.
Override with env var `CHROMA_PERSIST_DIR` or by passing args.
Collection name defaults to `tigergraph_docs`.
Override with env var `CHROMA_COLLECTION_NAME` or by passing args.

NOTE: this file is intentionally NOT named `chromadb.py` — a file with
that name shadows the installed `chromadb` package (`import chromadb`
would import itself) and breaks with:
`AttributeError: module 'chromadb' has no attribute 'PersistentClient'`.
"""

import os
from pathlib import Path

import chromadb

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Project root = parent of `database/` folder (i.e. D:\TigerGraph)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR", str(_PROJECT_ROOT / "chroma_db")
)
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "tigergraph_docs")
DEFAULT_DISTANCE = "cosine"  # good for normalized bge embeddings

__all__ = [
    "DEFAULT_PERSIST_DIR",
    "DEFAULT_COLLECTION_NAME",
    "client",
    "collection",
    "get_client",
    "get_collection",
    "init_chromadb",
    "reset_collection",
]


# ---------------------------------------------------------------------------
# Client / collection factories
# ---------------------------------------------------------------------------

def get_client(persist_directory: str = DEFAULT_PERSIST_DIR):
    """Initialize (or reuse) a persistent ChromaDB client backed by local disk.

    Args:
        persist_directory: Folder where ChromaDB stores its SQLite/Parquet
            files. Created automatically if missing.

    Returns:
        chromadb.PersistentClient instance.
    """
    persist_directory = os.path.abspath(os.path.expandvars(persist_directory))
    os.makedirs(persist_directory, exist_ok=True)
    return chromadb.PersistentClient(path=persist_directory)


def get_collection(
    name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    distance: str = DEFAULT_DISTANCE,
    client=None,
):
    """Establish a connection to a named collection (created if missing).

    Args:
        name: Collection name, e.g. "tigergraph_docs".
        persist_directory: Local persist dir for the PersistentClient.
        distance: HNSW space — "cosine" | "l2" | "ip".
        client: Optional pre-built PersistentClient (uses get_client() if None).

    Returns:
        chromadb Collection connected to the local persistent database.
    """
    if client is None:
        client = get_client(persist_directory)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": distance},
    )


def init_chromadb(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    distance: str = DEFAULT_DISTANCE,
):
    """Convenience: build client + named collection together.

    Returns:
        (client, collection) tuple.
    """
    _client = get_client(persist_directory)
    _collection = get_collection(
        name=collection_name,
        persist_directory=persist_directory,
        distance=distance,
        client=_client,
    )
    return _client, _collection


def reset_collection(
    name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    distance: str = DEFAULT_DISTANCE,
    client=None,
):
    """Delete (if exists) and recreate a named collection. Useful for re-ingest."""
    if client is None:
        client = get_client(persist_directory)
    try:
        client.delete_collection(name=name)
    except Exception:
        pass  # collection did not exist — fine
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": distance},
    )


# ---------------------------------------------------------------------------
# Eager initialization on import — so `from database.chroma_client import collection`
# gives you a ready-to-use local DB connection.
# ---------------------------------------------------------------------------

client, collection = init_chromadb()


if __name__ == "__main__":
    print(f"Persist dir : {os.path.abspath(DEFAULT_PERSIST_DIR)}")
    print(f"Collection  : {collection.name} (count={collection.count()})")
    print("ChromaDB local connection OK.")
