"""Chunk preprocessed Documents into token-bounded chunks.

Input format matches `ingestion/preprocessing.py::load_corpus`, i.e.
`Document(page_content="Title: {title}\\n\\n{text}", metadata={doc_id, title, url, wikidata_qid, approx_tokens})`.

Each chunk:
  1. Is split with `RecursiveCharacterTextSplitter` backed by tiktoken `cl100k_base`
     (chunk_size=512 tokens, chunk_overlap=54 tokens).
  2. Has the parent title prepended as `"Title: {title}\\n\\n..."` to preserve
     semantic context.
  3. Is returned as a `Document` in the same parent shape, with chunk metadata:
     {chunk_id, doc_id, title, source, wikidata_qid, chunk_index, token_count}
     (`url` and `approx_tokens` are also carried over for backwards compat).
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 54
TIKTOKEN_ENCODING = "cl100k_base"
TITLE_TEMPLATE = "Title: {title}\n\n{chunk_text}"

__all__ = [
    "CHUNK_SIZE_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "TIKTOKEN_ENCODING",
    "get_encoding",
    "count_tokens",
    "create_text_splitter",
    "chunk_document",
    "chunk_documents",
]


def get_encoding():
    """Return the tiktoken encoding used for length measurement."""
    import tiktoken

    return tiktoken.get_encoding(TIKTOKEN_ENCODING)


def count_tokens(text: str) -> int:
    """Count tokens in `text` with the configured tiktoken encoding."""
    return len(get_encoding().encode(text))


def create_text_splitter(
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
    encoding_name: str = TIKTOKEN_ENCODING,
) -> RecursiveCharacterTextSplitter:
    """Create a token-bounded RecursiveCharacterTextSplitter.

    Uses `from_tiktoken_encoder` so `chunk_size` / `chunk_overlap` are
    measured in tokens (not characters).
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _prepend_title(chunk_text: str, title: str) -> str:
    """Prepend `Title: {title}` header unless the chunk already has it."""
    title = (title or "").strip()
    if not title:
        return chunk_text.strip()
    chunk_text = chunk_text.strip()
    title_header = f"Title: {title}"
    if chunk_text.startswith(title_header):
        return chunk_text
    return TITLE_TEMPLATE.format(title=title, chunk_text=chunk_text)


def _build_chunk_document(
    chunk_text: str,
    parent: Document,
    chunk_index: int,
) -> Document:
    """Build one chunk Document in the parent format + chunk metadata."""
    parent_meta = parent.metadata or {}
    doc_id = parent_meta.get("doc_id", "")
    title = parent_meta.get("title", "")
    url = parent_meta.get("url", "")
    wikidata_qid = parent_meta.get("wikidata_qid")

    page_content = _prepend_title(chunk_text, title)

    metadata = {
        # Requested chunk fields
        "chunk_id": f"{doc_id}_chunk_{chunk_index}",
        "doc_id": doc_id,
        "title": title,
        "source": url,
        "wikidata_qid": wikidata_qid,
        "chunk_index": chunk_index,
        "token_count": count_tokens(page_content),
        # Carried over from parent for backwards compat
        "url": url,
    }
    if parent_meta.get("approx_tokens") is not None:
        metadata["approx_tokens"] = parent_meta.get("approx_tokens")

    return Document(page_content=page_content, metadata=metadata)


def chunk_document(
    document: Document,
    splitter: RecursiveCharacterTextSplitter | None = None,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[Document]:
    """Chunk a single parent Document, prepending the parent title to each chunk.

    To keep the *final* chunk (body + prepended title) within `chunk_size`
    tokens, the body is split with an effective size of
    `chunk_size - title_header_tokens`. A caller-supplied `splitter` is used
    as-is (final chunks may then slightly exceed `chunk_size` by the title
    length).
    """
    parent_meta = document.metadata or {}
    title = (parent_meta.get("title") or "").strip()

    if splitter is None:
        if title:
            header_tokens = count_tokens(f"Title: {title}\n\n")
            effective_size = max(50, chunk_size - header_tokens)
        else:
            effective_size = chunk_size
        splitter = create_text_splitter(
            chunk_size=effective_size, chunk_overlap=chunk_overlap
        )
    raw_chunks = splitter.split_text(document.page_content or "")
    chunks: list[Document] = []
    for i, raw_chunk in enumerate(raw_chunks):
        if not raw_chunk.strip():
            continue
        chunks.append(_build_chunk_document(raw_chunk, document, i))
    return chunks


def chunk_documents(
    documents: list[Document],
    splitter: RecursiveCharacterTextSplitter | None = None,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[Document]:
    """Chunk a list of parent Documents, preserving order.

    When `splitter` is None, each document gets a title-budget-adjusted
    splitter via :func:`chunk_document` so every final chunk stays within
    `chunk_size` tokens. Pass an explicit `splitter` to force one shared
    splitter for all docs.
    """
    if splitter is not None:
        all_chunks: list[Document] = []
        for doc in documents:
            all_chunks.extend(chunk_document(doc, splitter=splitter))
        return all_chunks
    all_chunks = []
    for doc in documents:
        all_chunks.extend(
            chunk_document(doc, splitter=None, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    return all_chunks
