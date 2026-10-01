"""Central model initializations.

Embedding model: BAAI/bge-small-en-v1.5 from Hugging Face (384-dim).
Initialized on import so callers can simply do:

    from ingestion.models import embedding_model
    # or
    from ingestion.models import get_embedding_model

Reranker model: BAAI/bge-reranker-base (CrossEncoder) from Hugging Face.
Loaded lazily on first use (large model) via:

    from ingestion.models import get_reranker_model
"""

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

__all__ = [
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_DIM",
    "RERANKER_MODEL_NAME",
    "embedding_model",
    "reranker_model",
    "init_embedding_model",
    "get_embedding_model",
    "init_reranker_model",
    "get_reranker_model",
]


def init_embedding_model(model_name: str = EMBEDDING_MODEL_NAME, **kwargs):
    """Initialize bge-small-en-v1.5 from Hugging Face.

    Prefers LangChain's HuggingFaceEmbeddings (works directly with
    embed_documents / embed_query), falls back to SentenceTransformer.
    """
    # Default: normalized embeddings (standard for bge models)
    kwargs.setdefault("encode_kwargs", {"normalize_embeddings": True})

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name, **kwargs)
    except ImportError:
        pass

    try:
        from sentence_transformers import SentenceTransformer

        # SentenceTransformer takes different kwargs; drop LangChain-only ones
        kwargs.pop("encode_kwargs", None)
        return SentenceTransformer(model_name, **kwargs)
    except ImportError:
        pass

    raise ImportError(
        "Could not load 'BAAI/bge-small-en-v1.5': install with "
        "`pip install -U langchain-huggingface sentence-transformers` "
        "(plus torch as needed for your platform)."
    )


# Eager initialization on import, as requested.
embedding_model = init_embedding_model()


def get_embedding_model(model=None):
    """Return the given model, or the module-level initialized model."""
    global embedding_model
    if model is not None:
        return model
    if embedding_model is None:
        embedding_model = init_embedding_model()
    return embedding_model


def init_reranker_model(model_name: str = RERANKER_MODEL_NAME, **kwargs):
    """Initialize the BAAI/bge-reranker-base CrossEncoder from Hugging Face.

    Uses `sentence_transformers.CrossEncoder` which takes (query, passage)
    pairs and returns relevance logits.
    """
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(model_name, **kwargs)
    except ImportError:
        pass

    raise ImportError(
        "Could not load 'BAAI/bge-reranker-base': install with "
        "`pip install -U sentence-transformers` "
        "(plus torch as needed for your platform)."
    )


# Lazy singleton: large model, so only loaded on first get_reranker_model().
reranker_model = None


def get_reranker_model(model=None):
    """Return the given model, or the lazily-initialized module reranker."""
    global reranker_model
    if model is not None:
        return model
    if reranker_model is None:
        reranker_model = init_reranker_model()
    return reranker_model
