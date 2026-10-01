"""Stage 4 — upsert resolved vertices/edges into TigerGraph in batches.

Wire format (verified against this TG 4.x instance — attributes sit directly
under the vertex/target ID as {"attr": {"value": ...}}, no wrapper):

    {"vertices": {vtype: {v_id: {"name": {"value": "..."}, ...}}},
     "edges":    {from_vtype: {from_id: {EDGE: {to_vtype: {to_id: {}}}}}}}

MENTIONED_IN edges need no attributes (the target chunk ID itself is the
provenance; the graph's edge types carry no attributes).
"""

import time

from tqdm import tqdm

from graphrag import config

__all__ = ["resolved_to_upserts", "load_resolved", "load_chunks"]

UPSERT_BATCH_OPERATIONS = 2000  # vertices+edges per REST call


def _vattrs(attrs: dict) -> dict:
    return {k: {"value": v} for k, v in (attrs or {}).items()}


def resolved_to_upserts(resolved: dict, chunk_records: list[dict] | None = None) -> list[dict]:
    """Convert resolve_extractions() output into per-batch upsert payloads.

    Args:
        resolved: Output of `graphrag.resolve.resolve_extractions`.
        chunk_records: Optional chunk dicts ({"id", "metadata"}) from Chroma;
            adds Chunk vertices so MENTIONED_IN edges always have a target.

    Returns:
        List of upsert payload dicts, each under the operation budget.
    """
    operations: list[tuple[str, ...]] = []  # ("v", vtype, vid, attrs) | ("e", fv, fid, en, tv, tid)

    vertices = {vtype: dict(vids) for vtype, vids in resolved["vertices"].items()}

    # Chunk vertices (only for chunks actually referenced by the resolved data).
    referenced_chunks = {
        e["to_id"] for e in resolved["edges"] if e["edge"] == "MENTIONED_IN"
    }
    vertices.setdefault("Chunk", {})
    for record in chunk_records or []:
        chunk_id = record.get("id") or record.get("metadata", {}).get("chunk_id")
        if chunk_id not in referenced_chunks or chunk_id in vertices["Chunk"]:
            continue
        meta = record.get("metadata", {})
        vertices["Chunk"][chunk_id] = {
            "attributes": {
                "doc_id": meta.get("doc_id", ""),
                "title": meta.get("title", ""),
                "chunk_index": str(meta.get("chunk_index", "0")),  # graph declares STRING
                "url": meta.get("url", meta.get("source", "")),
            }
        }

    for vtype, vids in vertices.items():
        for vid, data in vids.items():
            operations.append(("v", vtype, vid, data["attributes"]))

    for edge in resolved["edges"]:
        from_vtype = edge["from_type"] if edge["edge"] == "MENTIONED_IN" else config.EVENT_VERTEX
        operations.append(
            ("e", from_vtype, edge["from_id"], edge["edge"], edge["to_type"], edge["to_id"])
        )

    # Pack into batches.
    batches: list[dict] = []
    current: dict = {"vertices": {}, "edges": {}}
    count = 0

    def _flush():
        nonlocal current, count
        if count:
            batches.append(current)
        current, count = {"vertices": {}, "edges": {}}, 0

    for op in operations:
        if count >= UPSERT_BATCH_OPERATIONS:
            _flush()
        if op[0] == "v":
            _, vtype, vid, attrs = op
            current["vertices"].setdefault(vtype, {})[vid] = _vattrs(attrs)
        else:
            _, fv, fid, ename, tv, tid = op
            (
                current["edges"]
                .setdefault(fv, {})
                .setdefault(fid, {})
                .setdefault(ename, {})
                .setdefault(tv, {})
                .setdefault(tid, {})
            )
        count += 1
    _flush()
    return batches


def _count_accepted(result: dict, kind: str) -> int:
    """RESTPP returns snake_case or camelCase depending on version — accept both."""
    for key in (f"accepted_{kind}", f"accepted{kind.capitalize()}"):
        if isinstance(result, dict) and result.get(key) is not None:
            return int(result[key])
    return 0


def load_resolved(conn, resolved: dict, chunk_records: list[dict] | None = None) -> dict:
    """Upsert all batches; returns {"vertices_upserted", "edges_upserted", "calls"}."""
    batches = resolved_to_upserts(resolved, chunk_records)
    totals = {"vertices_upserted": 0, "edges_upserted": 0, "calls": 0}
    for payload in tqdm(batches, desc="tigergraph upsert"):
        result = conn.upsertData(payload)
        totals["vertices_upserted"] += _count_accepted(result, "vertices")
        totals["edges_upserted"] += _count_accepted(result, "edges")
        totals["calls"] += 1
        time.sleep(0.1)  # gentle on RESTPP
    return totals


def load_chunks(conn, chunk_records: list[dict]) -> int:
    """Upsert standalone Chunk vertices (no entity links) for the given chunks."""
    payload = {"vertices": {"Chunk": {}}, "edges": {}}
    for record in chunk_records:
        chunk_id = record.get("id") or record.get("metadata", {}).get("chunk_id")
        if not chunk_id:
            continue
        meta = record.get("metadata", {})
        payload["vertices"]["Chunk"][chunk_id] = _vattrs(
            {
                "doc_id": meta.get("doc_id", ""),
                "title": meta.get("title", ""),
                "chunk_index": str(meta.get("chunk_index", "0")),  # graph declares STRING
                "url": meta.get("url", meta.get("source", "")),
            }
        )
    result = conn.upsertData(payload)
    return _count_accepted(result, "vertices")
