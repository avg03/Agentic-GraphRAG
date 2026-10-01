"""Stage 3 — deduplicate/resolve extracted entities and events (no LLM calls).

Deterministic, cheap resolution:
  1. Normalize names (case, punctuation, leading articles, whitespace).
  2. Merge entity mentions with the same (type, normalized name) into one
     canonical vertex; keep the most frequent original casing as display name.
  3. Merge ActionEvents by normalized event name; union their relation lists
     (re-targeted through the alias map to canonical entity IDs).
  4. Every vertex records the chunk_ids it appeared in -> MENTIONED_IN edges.

This runs after extraction (cached JSONL) and before the TigerGraph load.
"""

from collections import Counter, defaultdict

from graphrag import config

__all__ = ["normalize_name", "resolve_extractions"]

_STOP_PREFIXES = ("the ", "a ", "an ")
_PUNCT = str.maketrans({c: " " for c in "\"'`.,;:!?()[]{}<>-/&*"})


def normalize_name(name: str) -> str:
    """Canonicalize an entity/event name for dedup keys."""
    lowered = (name or "").strip().lower().translate(_PUNCT)
    lowered = " ".join(lowered.split())
    for prefix in _STOP_PREFIXES:
        if lowered.startswith(prefix) and len(lowered) > len(prefix):
            lowered = lowered[len(prefix) :]
            break
    return lowered


def resolve_extractions(records: list[dict]) -> dict:
    """Merge extraction records into resolved graph data.

    Args:
        records: Extraction records (from extract_chunks or the cached JSONL);
            each has entities[], action_events[], chunk_id, doc_id.

    Returns:
        {
          "vertices": {vtype: {vid: {"attributes": {...},
                                     "mentioned_in": [chunk_id, ...]}}},
          "edges":    [{"edge", "from_id", "to_type", "to_id"}, ...],
          "aliases":  {original_name: (vtype, canonical_id)},
        }
        Vertex IDs are the normalized names (unique within a vertex type,
        which is all TigerGraph requires of a PKEY).
    """
    # ---------------- entities ----------------
    # key: (type, normalized) -> accumulation
    entity_acc: dict[tuple[str, str], dict] = {}
    aliases: dict[str, tuple[str, str]] = {}

    for record in records:
        chunk_id = record.get("chunk_id", "")
        for ent in record.get("entities", []):
            name = (ent.get("name") or "").strip()
            vtype = ent.get("type")
            if not name or vtype not in config.ENTITY_TYPES:
                continue
            norm = normalize_name(name)
            if not norm:
                continue
            key = (vtype, norm)
            acc = entity_acc.setdefault(
                key,
                {"names": Counter(), "descriptions": [], "mentioned_in": set()},
            )
            acc["names"][name] += 1
            if ent.get("description"):
                acc["descriptions"].append(ent["description"].strip())
            acc["mentioned_in"].add(chunk_id)
            # First type seen wins for a raw name (keeps mapping stable).
            aliases.setdefault(name, key)

    vertices: dict[str, dict[str, dict]] = defaultdict(dict)
    for (vtype, norm), acc in entity_acc.items():
        display_name = acc["names"].most_common(1)[0][0]
        attrs = {"name": display_name, "description": acc["descriptions"][0] if acc["descriptions"] else ""}
        if vtype == "Outcome":
            attrs["outcome_kind"] = ""
        vertices[vtype][norm] = {
            "attributes": attrs,
            "mentioned_in": sorted(acc["mentioned_in"]),
        }

    # ---------------- action events ----------------
    event_acc: dict[str, dict] = {}
    for record in records:
        chunk_id = record.get("chunk_id", "")
        for event in record.get("action_events", []):
            name = (event.get("name") or "").strip()
            norm = normalize_name(name)
            if not norm:
                continue
            acc = event_acc.setdefault(
                norm,
                {
                    "names": Counter(),
                    "action_types": Counter(),
                    "descriptions": [],
                    "impacts": [],
                    "mentioned_in": set(),
                    "relations": {e: set() for e in config.EVENT_EDGES},
                },
            )
            acc["names"][name] += 1
            if event.get("action_type"):
                acc["action_types"][event["action_type"]] += 1
            if event.get("description"):
                acc["descriptions"].append(event["description"].strip())
            if event.get("impact_summary"):
                acc["impacts"].append(event["impact_summary"].strip())
            acc["mentioned_in"].add(chunk_id)
            for edge_name, field in (
                ("INITIATED_BY", "initiated_by"),
                ("TOOK_PLACE_AT", "took_place_at"),
                ("AFFECTED", "affected"),
                ("RESULTED_IN", "resulted_in"),
            ):
                for raw_name in event.get(field, []):
                    target = aliases.get(raw_name)
                    if target is None:
                        continue
                    target_type, target_id = target
                    if target_type not in config.EVENT_EDGES[edge_name]:
                        continue  # force-map-or-drop
                    acc["relations"][edge_name].add((target_type, target_id))

    for norm, acc in event_acc.items():
        vertices["ActionEvent"][norm] = {
            "attributes": {
                "name": acc["names"].most_common(1)[0][0],
                "action_type": acc["action_types"].most_common(1)[0][0] if acc["action_types"] else "",
                "description": acc["descriptions"][0] if acc["descriptions"] else "",
                "impact_summary": acc["impacts"][0] if acc["impacts"] else "",
            },
            "mentioned_in": sorted(acc["mentioned_in"]),
        }

    # ---------------- edges ----------------
    edges: list[dict] = []
    for norm, acc in event_acc.items():
        for edge_name, targets in acc["relations"].items():
            for target_type, target_id in sorted(targets):
                edges.append(
                    {
                        "edge": edge_name,
                        "from_id": norm,
                        "to_type": target_type,
                        "to_id": target_id,
                    }
                )

    # MENTIONED_IN: every vertex -> every chunk it appeared in.
    for vtype, vids in vertices.items():
        if vtype == "Chunk":
            continue
        for vid, data in vids.items():
            for chunk_id in data["mentioned_in"]:
                if chunk_id:
                    edges.append(
                        {
                            "edge": "MENTIONED_IN",
                            "from_type": vtype,
                            "from_id": vid,
                            "to_type": "Chunk",
                            "to_id": chunk_id,
                        }
                    )

    return {"vertices": dict(vertices), "edges": edges, "aliases": aliases}
