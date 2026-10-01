"""BM25 ranking over raw chunk Documents (stdlib only, no extra deps).

Takes LangChain chunk Documents (see `ingestion/chunking.py`):
    Document(
        page_content="Title: {title}\\n\\n{chunk text}",
        metadata={chunk_id, doc_id, title, source, wikidata_qid,
                  chunk_index, token_count, url, ...},
    )

builds a token corpus from their `page_content`, scores a query with
Okapi BM25 (k1=1.5, b=0.75), and returns the top-m chunks.

Usage:
    from langchain_core.documents import Document
    from retrieval.bm25 import rank_chunks_bm25

    hits = rank_chunks_bm25("What is TigerGraph?", chunks, m=5)
    # [{"id", "document", "page_content", "metadata",
    #   "bm25_score", "rank"}, ...] best-first
"""

import math
import re

__all__ = [
    "tokenize",
    "get_corpus_texts",
    "build_bm25_corpus",
    "bm25_rank",
    "rank_chunks_bm25",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75


# ---------------------------------------------------------------------------
# Corpus building: chunks (Documents) -> token lists
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer (stdlib only)."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def get_corpus_texts(chunks) -> list[str]:
    """Extract the BM25 corpus texts from chunk Documents.

    Accepts langchain `Document` objects (uses `.page_content`).
    """
    texts: list[str] = []
    for c in chunks or []:
        if hasattr(c, "page_content"):
            texts.append(c.page_content or "")
        elif isinstance(c, dict) and "page_content" in c:
            texts.append(c.get("page_content") or "")
        elif isinstance(c, dict) and "document" in c:
            texts.append(c.get("document") or "")
        elif isinstance(c, str):
            texts.append(c)
        else:
            raise TypeError(
                "Unsupported chunk type for BM25 corpus: "
                f"expected Document/str/dict, got {type(c).__name__}."
            )
    return texts


def build_bm25_corpus(chunks) -> tuple[list[list[str]], list[int], dict[str, int], float]:
    """Tokenize corpus and precompute BM25 statistics.

    Returns:
        (tokenized_docs, doc_lens, doc_freqs, avgdl)
    """
    tokenized = [tokenize(t) for t in get_corpus_texts(chunks)]
    doc_lens = [len(d) for d in tokenized]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0
    doc_freqs: dict[str, int] = {}
    for doc in tokenized:
        for term in set(doc):
            doc_freqs[term] = doc_freqs.get(term, 0) + 1
    return tokenized, doc_lens, doc_freqs, avgdl


def _idf(term: str, doc_freqs: dict[str, int], n_docs: int) -> float:
    """Okapi IDF (always >= 0): log((N - df + 0.5) / (df + 0.5) + 1)."""
    df = doc_freqs.get(term, 0)
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def _score_one(
    query_terms: list[str],
    doc_terms: list[str],
    doc_len: int,
    doc_freqs: dict[str, int],
    n_docs: int,
    avgdl: float,
    k1: float = K1,
    b: float = B,
) -> float:
    """BM25 score of one document for tokenized query terms."""
    if not doc_terms or avgdl == 0:
        return 0.0
    tf: dict[str, int] = {}
    for t in doc_terms:
        tf[t] = tf.get(t, 0) + 1
    norm = k1 * (1.0 - b + b * doc_len / avgdl)
    score = 0.0
    for term in query_terms:
        f = tf.get(term, 0)
        if f == 0:
            continue
        score += _idf(term, doc_freqs, n_docs) * (f * (k1 + 1.0)) / (f + norm)
    return score


# ---------------------------------------------------------------------------
# Ranking: query + chunks -> top-m
# ---------------------------------------------------------------------------

def bm25_rank(
    query: str,
    chunks,
    m: int,
    k1: float = K1,
    b: float = B,
) -> list[dict]:
    """Rank raw chunk Documents with BM25 and return the top-m.

    Args:
        query: Raw query string (non-empty).
        chunks: List of chunk Documents (`page_content` + metadata).
        m: Number of top chunks to return.
        k1: BM25 term-frequency saturation (default 1.5).
        b: BM25 length-normalization (default 0.75).

    Returns:
        List (best-first, length min(m, N)) of dicts:
        {"id", "document", "page_content", "metadata",
         "bm25_score", "rank"}
        where `id` is `metadata.chunk_id` (fallback: index-based),
        `document`/`page_content` is the chunk text, and `metadata`
        is the chunk's metadata dict.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    chunks = list(chunks or [])
    if not chunks:
        return []
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer.")

    query_terms = tokenize(query)
    if not query_terms:
        return []

    tokenized, doc_lens, doc_freqs, avgdl = build_bm25_corpus(chunks)
    n_docs = len(chunks)

    scored: list[dict] = []
    for i, chunk in enumerate(chunks):
        score = _score_one(
            query_terms, tokenized[i], doc_lens[i],
            doc_freqs, n_docs, avgdl, k1=k1, b=b,
        )
        if hasattr(chunk, "page_content"):
            text = chunk.page_content or ""
            meta = dict(getattr(chunk, "metadata", None) or {})
        elif isinstance(chunk, dict):
            text = chunk.get("page_content", chunk.get("document", "")) or ""
            meta = dict(chunk.get("metadata", {}) or {})
        else:  # plain string chunk
            text = chunk
            meta = {}
        chunk_id = meta.get("chunk_id") or f"chunk_{i}"
        scored.append(
            {
                "id": chunk_id,
                "document": text,
                "page_content": text,
                "metadata": meta,
                "bm25_score": float(score),
            }
        )

    scored.sort(key=lambda h: h["bm25_score"], reverse=True)
    top = scored[: min(m, len(scored))]
    for rank, hit in enumerate(top, start=1):
        hit["rank"] = rank
    return top


# Backwards-friendly alias: same signature, explicit name.
def rank_chunks_bm25(query: str, chunks, m: int) -> list[dict]:
    """Alias of `bm25_rank`: return top-m chunks for `query`."""
    return bm25_rank(query, chunks, m)


if __name__ == "__main__":
    from langchain_core.documents import Document

    demo = [
        Document(
            page_content="Title: TigerGraph\n\nTigerGraph is a graph database.",
            metadata={"chunk_id": "D1_chunk_0", "doc_id": "D1"},
        ),
        Document(
            page_content="Title: Bananas\n\nBananas are yellow fruit.",
            metadata={"chunk_id": "D2_chunk_0", "doc_id": "D2"},
        ),
        Document(
            page_content="Title: GSQL\n\nTigerGraph supports GSQL graph queries.",
            metadata={"chunk_id": "D3_chunk_0", "doc_id": "D3"},
        ),
    ]
    for hit in rank_chunks_bm25("TigerGraph graph database", demo, m=2):
        print(f"[rank {hit['rank']}] {hit['id']} score={hit['bm25_score']:.4f}")
