"""Embed chunked Documents (see `ingestion/chunking.py`).

Chunk input format:
    Document(
        page_content="Title: {title}\\n\\n{chunk text}",
        metadata={chunk_id, doc_id, title, source, wikidata_qid,
                  chunk_index, token_count, url, ...},
    )

The model (`BAAI/bge-small-en-v1.5`, 384-dim) is connected
from `ingestion.models` on import. It can be swapped if needed:

    from ingestion import store_embed
    store_embed.set_embedding_model(my_model)  # or pass model=... per call

Supported model interfaces (checked in this order):
  1. SentenceTransformer-style: `model.encode(texts, ...)` / `model.encode(text)`
  2. LangChain-style: `model.embed_documents(texts)` / `model.embed_query(text)`

The default model is connected from `ingestion.models` (bge-small-en-v1.5,
initialized on import):

    from ingestion.store_embed import embed_chunks  # uses connected model
"""

from langchain_core.documents import Document

from .models import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    embedding_model,
    get_embedding_model as _get_central_model,
    init_embedding_model as _init_central_model,
)

__all__ = [
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_DIM",
    "embedding_model",
    "set_embedding_model",
    "get_embedding_model",
    "init_embedding_model",
    "embed_chunk",
    "embed_chunks",
]


def set_embedding_model(model) -> None:
    """Inject a replacement embedding model (updates models + this module)."""
    global embedding_model
    embedding_model = model
    try:
        from . import models as _models

        _models.embedding_model = model
    except ImportError:
        pass


def init_embedding_model(model_name: str = EMBEDDING_MODEL_NAME, **kwargs):
    """(Re)initialize bge-small-en-v1.5 via `ingestion.models`."""
    global embedding_model
    embedding_model = _init_central_model(model_name, **kwargs)
    return embedding_model


def get_embedding_model(model=None):
    """Resolve the model: explicit arg > connected `ingestion.models` model."""
    if model is not None:
        return model
    return _get_central_model()


def _embed_texts(texts: list[str], model) -> list[list[float]]:
    """Embed a batch of texts via SentenceTransformer- or LangChain-style model."""
    if hasattr(model, "encode"):
        # SentenceTransformer-style
        vectors = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vectors]
    if hasattr(model, "embed_documents"):
        # LangChain-style
        return [list(map(float, v)) for v in model.embed_documents(texts)]
    if hasattr(model, "embed_query") and len(texts) == 1:
        return [list(map(float, model.embed_query(texts[0])))]
    raise TypeError(
        "Unsupported embedding model interface: expected "
        "`encode(...)` (SentenceTransformer) or "
        "`embed_documents(...)` (LangChain). "
        f"Got {type(model).__name__}."
    )


def embed_chunk(chunk: Document, model=None) -> dict:
    """Embed one chunk Document's `page_content`; return embedding + metadata.

    Returns:
        {"chunk_id": str, "doc_id": str, "page_content": str,
         "metadata": dict, "embedding": list[float]}
    """
    resolved = get_embedding_model(model)
    text = chunk.page_content or ""
    if not text.strip():
        raise ValueError("Cannot embed empty chunk page_content.")
    embedding = _embed_texts([text], resolved)[0]
    meta = dict(chunk.metadata or {})
    return {
        "chunk_id": meta.get("chunk_id", ""),
        "doc_id": meta.get("doc_id", ""),
        "page_content": text,
        "metadata": meta,
        "embedding": embedding,
    }


def embed_chunks(
    chunks: list[Document],
    model=None,
    batch_size: int = 32,
) -> list[dict]:
    """Embed a list of chunk Documents in batches; preserve input order.

    Returns a list of `embed_chunk`-style dicts, one per input chunk.
    """
    resolved = get_embedding_model(model)
    results: list[dict] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content or "" for c in batch]
        if any(not t.strip() for t in texts):
            raise ValueError("Cannot embed chunk with empty page_content.")
        vectors = _embed_texts(texts, resolved)
        for chunk, vector in zip(batch, vectors):
            meta = dict(chunk.metadata or {})
            results.append(
                {
                    "chunk_id": meta.get("chunk_id", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "page_content": chunk.page_content,
                    "metadata": meta,
                    "embedding": vector,
                }
            )
    return results
