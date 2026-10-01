"""Query-side helpers: clean a raw string query + embed it.

Pipeline:
    raw query string
      -> clean_query()          (HTML, emojis, punctuation except . , )
      -> embed_query_text()     (BAAI/bge-small-en-v1.5 via ingestion.models)
      -> clean_and_embed_query() (convenience: both steps at once)

Usage:
    from retrieval.query import clean_query, embed_query_text, clean_and_embed_query

    cleaned = clean_query("<p>What is TigerGraph??! 😀</p>")
    # -> "what is tigergraph."

    vec = embed_query_text("What is TigerGraph?")
    out = clean_and_embed_query("What is TigerGraph??! 😀")
    # -> {"cleaned_query": ..., "embedding": [...]}
"""

import html
import re

__all__ = [
    "clean_query",
    "embed_query_text",
    "clean_and_embed_query",
]

# ---------------------------------------------------------------------------
# 1. Cleaning
# ---------------------------------------------------------------------------

# HTML tags: <...> (non-greedy, no nesting handling needed for queries)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Emoji / pictograph / symbol ranges + variation selectors + ZWJ + flags.
# Covers: emoticons, dingbats, transport/map symbols, enclosed chars,
# regional indicators (flags), skin-tone modifiers, VS16, ZWJ sequences.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # emoticons, pictographs, symbols, transport, etc.
    "\u2600-\u27BF"          # misc symbols, dingbats, arrows
    "\u2B00-\u2BFF"          # misc symbols and arrows
    "\uFE00-\uFE0F"          # variation selectors
    "\u200D"                 # zero-width joiner (joins emoji sequences)
    "\u2190-\u21FF"          # arrows
    "\u2300-\u23FF"          # misc technical
    "\u25A0-\u25FF"          # geometric shapes
    "\u2700-\u27BF"          # dingbats (overlap, kept explicit)
    "\U0001F1E6-\U0001F1FF"  # regional indicator symbols (flags)
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
    "]+",
    flags=re.UNICODE,
)

# Keep ONLY: a-z, 0-9, whitespace, period, comma.
# Everything else is replaced with a space (avoids merging words:
# "hello#world" -> "hello world", not "helloworld").
_DISALLOWED_RE = re.compile(r"[^a-z0-9\s.,]")


def clean_query(query: str) -> str:
    """Clean a raw query string.

    Steps (in order):
      1. Decode HTML entities (e.g. ``&amp;`` -> ``&``).
      2. Remove HTML tags (``<p>hi</p>`` -> ``hi``).
      3. Remove emojis / pictographs / symbols.
      4. Lowercase.
      5. Remove irrelevant punctuation: keep ONLY ``.`` and ``,``;
         every other non-alphanumeric char becomes a space.
      6. Collapse whitespace to single spaces and strip.

    Args:
        query: Raw query string.

    Returns:
        Cleaned query string (possibly empty if nothing remains).

    Raises:
        TypeError: If ``query`` is not a string.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}.")

    # 1. Decode entities first so e.g. "&lt;b&gt;" becomes "<b>" and is
    #    then stripped as a tag in step 2.
    text = html.unescape(query)

    # 2. Strip HTML tags -> space (avoid merging words across tags).
    text = _HTML_TAG_RE.sub(" ", text)

    # 3. Strip emojis / symbols.
    text = _EMOJI_RE.sub("", text)

    # 4. Lowercase (light-clean, matches retrieval convention).
    text = text.lower()

    # 5. Keep only a-z, 0-9, whitespace, period, comma.
    text = _DISALLOWED_RE.sub(" ", text)

    # 6. Collapse whitespace and strip.
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# 2. Embedding (reuses ingestion.models: BAAI/bge-small-en-v1.5, 384-dim)
# ---------------------------------------------------------------------------

def embed_query_text(query: str, model=None, clean: bool = True) -> list[float]:
    """Embed a query string into a vector.

    Reuses the central model from ``ingestion.models``
    (``BAAI/bge-small-en-v1.5``, 384-dim, normalized) unless an
    override ``model`` is passed. Supports, in order:
      1. SentenceTransformer-style: ``model.encode(...)``
      2. LangChain-style: ``model.embed_query(...)`` / ``model.embed_documents(...)``

    Args:
        query: Raw (or pre-cleaned) query string. Must be non-empty
            after optional cleaning.
        model: Optional embedding-model override. Defaults to
            ``ingestion.models.get_embedding_model()``.
        clean: If True (default), run :func:`clean_query` first.
            Pass ``clean=False`` if the input is already cleaned.

    Returns:
        Embedding as a list of floats (384-dim for bge-small-en-v1.5).

    Raises:
        ValueError: If the (cleaned) query is empty.
        TypeError: If the model interface is unsupported.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}.")

    text = clean_query(query) if clean else query.strip()
    if not text:
        raise ValueError("query must be a non-empty string (after cleaning).")

    # Lazy import so this module stays importable without torch/HF installed
    # until an embedding is actually requested.
    from ingestion.models import get_embedding_model

    resolved = get_embedding_model(model)

    if hasattr(resolved, "encode"):
        # SentenceTransformer-style (normalized like ingestion path)
        vec = resolved.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )
        return [float(x) for x in vec]

    if hasattr(resolved, "embed_query"):
        # LangChain-style single-query path (preferred for queries)
        return [float(x) for x in resolved.embed_query(text)]

    if hasattr(resolved, "embed_documents"):
        return [float(x) for x in resolved.embed_documents([text])[0]]

    raise TypeError(
        "Unsupported embedding model interface: expected "
        "`encode(...)` (SentenceTransformer) or "
        "`embed_query(...)`/`embed_documents(...)` (LangChain). "
        f"Got {type(resolved).__name__}."
    )


# ---------------------------------------------------------------------------
# 3. Combined helper
# ---------------------------------------------------------------------------

def clean_and_embed_query(query: str, model=None) -> dict:
    """Clean ``query`` and embed the cleaned form in one call.

    Args:
        query: Raw query string.
        model: Optional embedding-model override (see :func:`embed_query_text`).

    Returns:
        ``{"cleaned_query": str, "embedding": list[float]}``.

    Raises:
        TypeError / ValueError: Propagated from cleaning / embedding
        (e.g. empty query after cleaning).
    """
    cleaned = clean_query(query)
    if not cleaned:
        raise ValueError("query must be a non-empty string (after cleaning).")
    embedding = embed_query_text(cleaned, model=model, clean=False)
    return {"cleaned_query": cleaned, "embedding": embedding}


if __name__ == "__main__":
    demo = "<p>What is TigerGraph??! 😀 &amp; how does it work...</p>"
    print(f"Raw:     {demo!r}")
    print(f"Cleaned: {clean_query(demo)!r}")
    try:
        out = clean_and_embed_query(demo)
        print(f"Embedding dim: {len(out['embedding'])}")
        print(f"First 5 vals:  {out['embedding'][:5]}")
    except Exception as e:  # model deps may be missing
        print(f"(Skipping embedding demo: {e})")
