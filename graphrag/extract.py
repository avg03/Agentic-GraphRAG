"""Stage 2 — entity/relation extraction per chunk via Gemini (structured output).

Input : chunk dicts read back from ChromaDB ({"id", "document"/"page_content",
        "metadata"}) — chunks are NOT re-created here (ingestion runs once).
Output: per-doc JSONL checkpoint files `graphrag/extracted/{doc_id}.jsonl`,
        one line per chunk: ChunkExtraction fields + chunk_id + doc_id.

Checkpointing means re-runs only call Gemini for chunks not yet extracted.

Usage:
    from graphrag.extract import extract_chunks

    results, metrics = extract_chunks(chunks, concurrency=4)
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from graphrag import config
from shared.llm_client import generate_structured
from shared.metrics import Metrics
from shared.models import ChunkExtraction
from shared.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt

__all__ = ["extract_chunks", "load_cached_extractions"]

_lock = threading.Lock()


def _cache_path(chunk_id: str) -> Path:
    doc_id = chunk_id.rsplit("_chunk_", 1)[0]
    return config.EXTRACTED_DIR / f"{doc_id}.jsonl"


def _load_cached(cache_file: Path) -> dict[str, dict]:
    """Read a doc's checkpoint file -> {chunk_id: extraction_dict}."""
    if not cache_file.exists():
        return {}
    cached = {}
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            cached[record["chunk_id"]] = record
    return cached


def _extract_one(chunk: dict) -> tuple[str, dict]:
    """Extract one chunk through Gemini; returns (chunk_id, record)."""
    chunk_id = chunk.get("id") or chunk["metadata"].get("chunk_id", "")
    text = chunk.get("document") or chunk.get("page_content") or ""
    title = chunk["metadata"].get("title", "")

    prompt = build_extraction_prompt(title=title, chunk_text=text)
    parsed, usage = generate_structured(
        prompt=prompt,
        response_schema=ChunkExtraction,
        system=EXTRACTION_SYSTEM_PROMPT,
    )

    clean, dropped = config.validate_extraction_against_ontology(parsed.model_dump())
    record = {
        **clean,
        "chunk_id": chunk_id,
        "doc_id": chunk["metadata"].get("doc_id", ""),
        "_ontology_dropped": dropped,
    }
    return chunk_id, record


def extract_chunks(
    chunks: list[dict],
    concurrency: int | None = None,
    metrics: Metrics | None = None,
) -> list[dict]:
    """Extract entities/action-events for every chunk (cached where possible).

    Args:
        chunks: Chunk dicts as returned by Chroma `collection.get(...)`
            (keys: id, document, metadata) — see ingest_pipeline.load_chunks.
        concurrency: Parallel Gemini calls (default config.EXTRACTION_CONCURRENCY).
        metrics: Optional Metrics to accumulate LLM usage + counters.

    Returns:
        One extraction record dict per chunk (order follows input).
    """
    concurrency = concurrency or config.EXTRACTION_CONCURRENCY
    metrics = metrics or Metrics()
    config.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    # Load per-doc caches once.
    cache_by_doc: dict[str, dict[str, dict]] = {}
    for chunk in chunks:
        doc_id = chunk["metadata"].get("doc_id", "") or "_"
        if doc_id not in cache_by_doc:
            cache_by_doc[doc_id] = _load_cached(_cache_path(chunk["id"]))

    pending = [
        c
        for c in chunks
        if c["metadata"].get("chunk_id", c.get("id")) not in cache_by_doc.get(
            c["metadata"].get("doc_id", "") or "_", {}
        )
    ]
    metrics.incr("chunks_cached", len(chunks) - len(pending))
    metrics.incr("chunks_processed", len(pending))

    results: dict[str, dict] = {}
    for chunk in chunks:
        doc_id = chunk["metadata"].get("doc_id", "") or "_"
        cached = cache_by_doc[doc_id].get(chunk["metadata"].get("chunk_id", chunk.get("id")))
        if cached is not None:
            results[chunk["id"]] = cached

    # Concurrent extraction for the rest.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_extract_one, chunk): chunk for chunk in pending}
        with tqdm(total=len(futures), desc="gemini extract") as bar:
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    chunk_id, record = future.result()
                except Exception as exc:
                    metrics.incr("chunks_failed")
                    bar.update(1)
                    tqdm.write(f"FAILED chunk {chunk.get('id')}: {exc}")
                    continue
                metrics.incr("ontology_dropped", record.get("_ontology_dropped", 0))
                metrics.incr("entities", len(record.get("entities", [])))
                metrics.incr("action_events", len(record.get("action_events", [])))

                results[chunk_id] = record
                # Append to the doc's checkpoint file (thread-safe).
                cache_file = _cache_path(chunk_id)
                with _lock:
                    with open(cache_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                bar.update(1)

    return [results[c["id"]] for c in chunks if c["id"] in results]


def load_cached_extractions(docs: list[str] | None = None) -> list[dict]:
    """Load extraction records from the checkpoint dir (optionally for given doc_ids)."""
    records: list[dict] = []
    for cache_file in sorted(config.EXTRACTED_DIR.glob("*.jsonl")):
        if docs is not None and cache_file.stem not in docs:
            continue
        records.extend(_load_cached(cache_file).values())
    return records
