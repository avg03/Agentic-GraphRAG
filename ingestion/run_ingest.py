"""One-time corpus ingestion: JSONL corpus -> chunks -> embeddings -> ChromaDB.

This is the SINGLE ingestion entry point for the whole project (per design:
ingestion runs once; all three pipelines read the same ChromaDB collection,
no pipeline re-chunks).

Steps (all reuse existing modules):
    1. load_corpus        (ingestion.preprocessing)  Data/corpus.jsonl -> Documents
    2. chunk_documents    (ingestion.chunking)       512-token chunks with stable
                                                     chunk_id = {doc_id}_chunk_{i}
    3. embed_chunks       (ingestion.store_embed)    bge-small-en-v1.5 vectors
    4. collection.add     (database.chroma_client)   persist to local ChromaDB

Usage:
    python -m ingestion.run_ingest --n-docs 150          # pilot: first 150 docs
    python -m ingestion.run_ingest --full                # entire corpus
    python -m ingestion.run_ingest --reset --full        # wipe collection first

Already-ingested chunk_ids are skipped on re-run (idempotent top-up).
"""

import argparse

from tqdm import tqdm

from database.chroma_client import get_collection, reset_collection
from ingestion.chunking import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, chunk_documents
from ingestion.preprocessing import load_corpus
from ingestion.store_embed import embed_chunks

__all__ = ["ingest_corpus", "main"]

DEFAULT_CORPUS_PATH = "Data/corpus.jsonl"
CHROMA_ADD_BATCH = 128


def _sanitize_metadata(meta: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool — drop None/other."""
    allowed = (str, int, float, bool)
    return {k: v for k, v in meta.items() if isinstance(v, allowed)}


def ingest_corpus(
    corpus_path: str = DEFAULT_CORPUS_PATH,
    n_docs: int | None = None,
    reset: bool = False,
    collection=None,
    batch_size: int = CHROMA_ADD_BATCH,
) -> dict:
    """Ingest (a subset of) the corpus into ChromaDB. Returns a summary dict.

    Args:
        corpus_path: Path to JSONL corpus.
        n_docs: Ingest only the first N documents (pilot). None = all.
        reset: Delete and recreate the collection before ingesting.
        collection: Optional Chroma collection override (default: local
            ``tigergraph_docs`` collection).
        batch_size: Chroma add batch size.

    Returns:
        {"docs_loaded", "chunks_created", "chunks_new", "chunks_skipped",
         "collection_count"}
    """
    if collection is None:
        collection = get_collection()

    if reset:
        print(f"Resetting collection {collection.name!r} ...")
        collection = reset_collection()

    # 1. Load corpus
    documents = load_corpus(corpus_path)
    if n_docs is not None:
        documents = documents[:n_docs]
    print(f"Loaded {len(documents)} documents from {corpus_path}")

    # 2. Chunk
    chunks = chunk_documents(documents)
    print(
        f"Created {len(chunks)} chunks "
        f"({CHUNK_SIZE_TOKENS} tokens / {CHUNK_OVERLAP_TOKENS} overlap)"
    )

    # 3. Skip chunks already in the collection (idempotent re-runs)
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in existing_ids]
    skipped = len(chunks) - len(new_chunks)
    if skipped:
        print(f"Skipping {skipped} chunks already in collection")

    # 4. Embed + add in batches
    added = 0
    for start in tqdm(range(0, len(new_chunks), batch_size), desc="embed+add"):
        batch = new_chunks[start : start + batch_size]
        embedded = embed_chunks(batch)
        collection.add(
            ids=[e["chunk_id"] for e in embedded],
            documents=[e["page_content"] for e in embedded],
            embeddings=[e["embedding"] for e in embedded],
            metadatas=[_sanitize_metadata(e["metadata"]) for e in embedded],
        )
        added += len(embedded)

    summary = {
        "docs_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunks_new": added,
        "chunks_skipped": skipped,
        "collection_count": collection.count(),
    }
    print(
        "Done: {chunks_new} new chunks added; collection now has "
        "{collection_count} chunks.".format(**summary)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS_PATH, help="JSONL corpus path")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--n-docs", type=int, default=None, help="Ingest first N docs only (pilot)")
    group.add_argument("--full", action="store_true", help="Ingest the entire corpus")
    parser.add_argument("--reset", action="store_true", help="Wipe the collection before ingesting")
    parser.add_argument("--batch-size", type=int, default=CHROMA_ADD_BATCH)
    args = parser.parse_args()

    n_docs = None if args.full else args.n_docs
    if n_docs is None and not args.full:
        parser.error("specify --n-docs N (pilot) or --full")

    ingest_corpus(
        corpus_path=args.corpus,
        n_docs=n_docs,
        reset=args.reset,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
