"""Backfill structured context attributes onto ActionEvent vertices (v2 §3.2).

The optional final step of the v2 migration: pushes the deterministically
derived context metadata (root_context / context_level / context_path /
context_keys) into TigerGraph so the graph itself carries scope provenance.
Retrieval NEVER depends on this — the context index is local — but the
attributes enable graph-side scope queries and the future HAS_EVENT
extension (v2 §20).

Steps:
    1. connect + verify
    2. ensure the 4 context attributes exist (idempotent SCHEMA_CHANGE job)
    3. re-resolve cached extractions with Chroma titles (zero LLM calls)
    4. batch-upsert ONLY the context attributes for every ActionEvent

Usage:
    python -m graphrag.backfill_context --dry-run   # preview counts only
    python -m graphrag.backfill_context             # full backfill
"""

import argparse
import json
import time

from tqdm import tqdm

__all__ = ["run_backfill"]


def run_backfill(dry_run: bool = False, batch_ops: int = 2000) -> dict:
    from database.chroma_client import get_collection
    from graphrag import config
    from graphrag.context import context_keys_from_attrs
    from graphrag.extract import load_cached_extractions
    from graphrag.ingest_pipeline import load_chunks_from_chroma
    from graphrag.resolve import resolve_extractions
    from graphrag.schema import connect, ensure_context_attributes, verify_connection

    conn = connect()
    verify_connection(conn)
    schema_info = ensure_context_attributes(conn, dry_run=dry_run)
    print(f"schema: added={schema_info['added']} present={schema_info['present']}")
    if dry_run:
        return schema_info

    # attributes now exist -> stop load_graph from stripping them
    config.CONTEXT_ATTRS_IN_GRAPH = True

    records = load_cached_extractions()
    if not records:
        raise RuntimeError("No cached extractions — nothing to backfill.")
    titles = {
        c["metadata"].get("doc_id", ""): c["metadata"].get("title", "")
        for c in load_chunks_from_chroma(collection=get_collection())
        if c.get("metadata", {}).get("doc_id")
    }
    resolved = resolve_extractions(records, titles=titles)
    events = resolved["vertices"].get("ActionEvent", {})
    print(f"resolved {len(events)} ActionEvents")

    # payload: context attributes only (never touch the v1 attributes)
    ops = 0
    payload = {"vertices": {config.EVENT_VERTEX: {}}, "edges": {}}
    batches = []

    def _flush():
        nonlocal payload, ops
        if ops:
            batches.append(payload)
        payload = {"vertices": {config.EVENT_VERTEX: {}}, "edges": {}}
        ops = 0

    for vid, data in events.items():
        attrs = data.get("attributes", {})
        if not context_keys_from_attrs(attrs):
            continue  # no context derived — nothing to write
        payload["vertices"][config.EVENT_VERTEX][vid] = {
            k: {"value": attrs[k]}
            for k in config.CONTEXT_ATTRS
            if k in attrs
        }
        ops += 1
        if ops >= batch_ops:
            _flush()
    _flush()

    totals = {"vertices": 0, "calls": 0}
    for batch in tqdm(batches, desc="context backfill"):
        result = conn.upsertData(batch)
        accepted = result.get("accepted_vertices", result.get("acceptedVertices", 0)) if isinstance(result, dict) else 0
        totals["vertices"] += int(accepted or 0)
        totals["calls"] += 1
        time.sleep(0.1)

    print(f"backfilled context attributes on {totals['vertices']} vertices ({totals['calls']} calls)")
    schema_info["backfilled"] = totals
    return schema_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Schema preview only, no upserts")
    parser.add_argument("--batch-ops", type=int, default=2000)
    args = parser.parse_args()
    info = run_backfill(dry_run=args.dry_run, batch_ops=args.batch_ops)
    print(json.dumps({k: v for k, v in info.items() if k != "job"}, indent=1, default=str))


if __name__ == "__main__":
    main()
