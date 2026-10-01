"""Graph traversal: seed entities -> supporting chunks with support scores.

Primary path: REST `/graph/{g}/edges/{type}/{id}` per seed (returns every
edge touching the vertex in BOTH directions, so all four event edge families
are traversable). The installed-GSQL path is blocked by missing reverse
edges for AFFECTED/RESULTED_IN — see graphrag/install_queries.py docstring.

Support score for a chunk:
    +1 for each matched seed entity that links to it directly (MENTIONED_IN)
    +1 for each distinct ActionEvent that links both to the chunk and to a
       matched seed (event-mediated evidence)

Returns: {chunk_id: {"support": int, "via": [explanation strings]}}
"""

import time
from urllib.parse import quote

from graphrag import config

__all__ = ["traverse", "traverse_rest", "traverse_installed"]

QUERY_NAME = "graphrag_traverse"


def traverse(conn, seeds: list[tuple[str, str, str]]) -> dict:
    """Traverse from seeds via the REST edge endpoints (all edge types)."""
    if not seeds:
        return {}
    return traverse_rest(conn, seeds)


def traverse_rest(conn, seeds: list[tuple[str, str, str]]) -> dict:
    """REST fallback: per-seed edge queries (verified working endpoint).

    `/graph/{g}/edges/{type}/{id}` returns every edge touching the vertex
    (both directions), so one call per seed + one per related event suffices.
    """
    chunks: dict[str, dict] = {}
    url = conn.restppUrl + f"/graph/{conn.graphname}/edges/"

    def _get_edges(vtype: str, vid: str) -> list[dict]:
        try:
            res = conn._get(url + f"{quote(vtype)}/{quote(vid)}")
        except Exception:
            return []
        edges = res if isinstance(res, list) else res.get("results", [])
        seen, unique = set(), []
        for e in edges:
            key = (e.get("e_type"), e.get("from_id"), e.get("to_id"))
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    for vtype, vid, name in seeds:
        # 1. direct grounding: seed --MENTIONED_IN--> chunk (capped: broad
        #    seeds like 'hungary' can fan out to thousands of chunks and
        #    dilute the final context)
        per_seed = 0
        for e in _get_edges(vtype, vid):
            if e.get("e_type") == "MENTIONED_IN":
                if per_seed >= 25:
                    break
                cid = e["to_id"]
                entry = chunks.setdefault(cid, {"support": 0, "via": []})
                entry["support"] += 1
                entry["via"].append(f"direct:{name}")
                per_seed += 1

        # 2. event-mediated: seed <--> ActionEvent --MENTIONED_IN--> chunk
        event_ids = set()
        for e in _get_edges(vtype, vid):
            if e.get("to_type") == "ActionEvent":
                event_ids.add(e["to_id"])
            elif e.get("from_type") == "ActionEvent" and e.get("e_type") != "MENTIONED_IN":
                event_ids.add(e["from_id"])
        for eid in event_ids:
            for ce in _get_edges("ActionEvent", eid):
                if ce.get("e_type") == "MENTIONED_IN":
                    cid = ce["to_id"]
                    entry = chunks.setdefault(cid, {"support": 0, "via": []})
                    entry["support"] += 1
                    entry["via"].append(f"event:{eid[:48]}")
        time.sleep(0.05)

    return chunks


def traverse_installed(conn, seeds: list[tuple[str, str, str]]) -> dict:
    """Run the installed graphrag_traverse query (one round trip).

    Query emits flat string sets:
        direct_hits: "seed_entity_id|chunk_id"
        event_hits:  "event_id|chunk_id"
    Support per chunk = #distinct seeds (direct) + #distinct events (mediated).
    """
    params = {"seeds": [f"{vtype}:{vid}" for vtype, vid, _ in seeds]}
    result = conn.runInstalledQuery(QUERY_NAME, params=params)

    direct: set[tuple[str, str]] = set()
    events: set[tuple[str, str]] = set()
    for top in result:
        for key, values in top.items():
            if key == "top" or not isinstance(values, list):
                continue
            target = direct if "direct_hits" in key else events if "event_hits" in key else None
            if target is None:
                continue
            for raw in values:
                if isinstance(raw, str) and "|" in raw:
                    src, cid = raw.split("|", 1)
                    target.add((src, cid))

    chunks: dict[str, dict] = {}
    for src, cid in direct:
        entry = chunks.setdefault(cid, {"support": 0, "via": []})
        entry["support"] += 1
        entry["via"].append(f"direct:{src}")
    for src, cid in events:
        entry = chunks.setdefault(cid, {"support": 0, "via": []})
        entry["support"] += 1
        entry["via"].append(f"event:{src[:48]}")
    return chunks


def collect_relationships(conn, seeds: list[tuple[str, str, str]], max_paths: int = 20) -> list[str]:
    """Collect human-readable relationship paths around the seeds (1 hop).

    Reuses the same REST edge endpoints as traverse_rest; every edge touching
    a seed becomes a path string "Type1 -EDGE-> Type2". Event-mediated paths
    ("Event -EDGE-> Target") come from one extra hop around adjacent events.

    Returns at most `max_paths` unique path strings, preferring paths that
    involve an ActionEvent hub.
    """
    url = conn.restppUrl + f"/graph/{conn.graphname}/edges/"
    paths: list[str] = []
    seen: set[str] = set()
    event_ids: set[tuple[str, str]] = set()

    def _label(vtype: str, vid: str) -> str:
        short = vid if len(vid) <= 44 else vid[:44] + "…"
        return f"{vtype}:{short}"

    def _add(fr: str, edge: str, to: str) -> None:
        p = f"{fr} -{edge}-> {to}"
        if p not in seen:
            seen.add(p)
            paths.append(p)

    def _edges(vtype: str, vid: str) -> list[dict]:
        try:
            res = conn._get(url + f"{quote(vtype)}/{quote(vid)}")
        except Exception:
            return []
        edges = res if isinstance(res, list) else res.get("results", [])
        uniq, seen_keys = [], set()
        for e in edges:
            key = (e.get("e_type"), e.get("from_type"), e.get("from_id"), e.get("to_id"))
            if key not in seen_keys:
                seen_keys.add(key)
                uniq.append(e)
        return uniq

    for vtype, vid, _name in seeds:
        for e in _edges(vtype, vid):
            etype = e.get("e_type", "")
            if etype in ("reverse_AFFECTED", "reverse_INITIATED_BY",
                         "reverse_RESULTED_IN", "reverse_TOOK_PLACE_AT"):
                # in-edge from an event: normalize to forward direction
                _add(_label("ActionEvent", e["from_id"]), etype.replace("reverse_", ""),
                     _label(vtype, vid))
                event_ids.add(("ActionEvent", e["from_id"]))
            else:
                _add(_label(vtype, vid), etype, _label(e.get("to_type", ""), e["to_id"]))
                if e.get("to_type") == "ActionEvent":
                    event_ids.add(("ActionEvent", e["to_id"]))

    for _evt, eid in list(event_ids)[:5]:
        for e in _edges("ActionEvent", eid):
            _add(_label("ActionEvent", eid), e.get("e_type", ""),
                 _label(e.get("to_type", ""), e["to_id"]))
        if len(paths) >= max_paths * 2:
            break

    # Prefer event-centric paths first, then cap.
    event_first = [p for p in paths if p.startswith("ActionEvent:")]
    rest = [p for p in paths if not p.startswith("ActionEvent:")]
    return (event_first + rest)[:max_paths]
