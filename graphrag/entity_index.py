"""Entity index: graph vertex catalog + embeddings for query-entity matching.

Builds (and caches to disk) a catalog of every non-Chunk vertex:
    {type, id, name, description} + a BGE embedding of "name — description".

Matching a query mention:
    1. normalized exact name match (fast path, reuse resolve.normalize_name)
    2. otherwise cosine similarity >= SIM_THRESHOLD against the catalog
       (top-3 candidates)

The cache is rebuilt automatically when the graph's vertex count changes,
or forced with refresh=True.
"""

import json
from pathlib import Path

import numpy as np

from graphrag import config
from graphrag.resolve import normalize_name
from shared.llm_client import get_client  # noqa: F401  (ensures env loaded)

__all__ = [
    "build_entity_index",
    "load_entity_index",
    "match_entities",
    "SIM_THRESHOLD",
]

SIM_THRESHOLD = 0.75
_INDEX_DIR = Path(__file__).resolve().parent
_CATALOG_FILE = _INDEX_DIR / "entity_index.json"
_EMBED_FILE = _INDEX_DIR / "entity_index.npz"


def _fetch_vertices(conn) -> list[dict]:
    """Fetch all non-Chunk vertices from TigerGraph."""
    catalog = []
    for vtype in config.ALL_VERTEX_TYPES:
        if vtype == "Chunk":
            continue
        vertices = conn.getVertices(vtype, limit=100_000)
        for v in vertices:
            attrs = v.get("attributes", {})
            catalog.append(
                {
                    "type": vtype,
                    "id": v["v_id"],
                    "name": attrs.get("name", v["v_id"]),
                    "description": attrs.get("description", ""),
                }
            )
    return catalog


def _embed_texts(texts: list[str], model=None) -> np.ndarray:
    """Embed with the project's BGE embedder (ingestion.models)."""
    from ingestion.models import get_embedding_model

    resolved = get_embedding_model(model)
    if hasattr(resolved, "embed_documents"):
        vectors = resolved.embed_documents(texts)
    else:
        vectors = resolved.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-10, None)


def build_entity_index(conn, model=None, force: bool = False) -> dict:
    """(Re)build and cache the entity index. Returns {vertices, counts}."""
    catalog = _fetch_vertices(conn)
    if not catalog:
        raise RuntimeError("No vertices found — is the graph loaded?")

    texts = [
        f"{v['name']} — {v['description']}" if v["description"] else v["name"]
        for v in catalog
    ]
    embeddings = _embed_texts(texts, model)

    np.savez_compressed(_EMBED_FILE, embeddings=embeddings)
    meta = {
        "vertex_count": len(catalog),
        "graphname": config.TG_GRAPHNAME,
        "vertices": catalog,
        "norm_index": {normalize_name(v["name"]): i for i, v in enumerate(catalog)},
    }
    _CATALOG_FILE.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"Entity index built: {len(catalog)} vertices -> {_CATALOG_FILE.name}")
    return meta


def load_entity_index(conn=None, model=None, force: bool = False) -> dict:
    """Load the cached index, rebuilding it if stale or missing."""
    if not force and _CATALOG_FILE.exists() and _EMBED_FILE.exists():
        meta = json.loads(_CATALOG_FILE.read_text(encoding="utf-8"))
        if conn is not None:
            # Rebuild when the graph has grown since the cache was made.
            try:
                live = sum(
                    conn.getVertexCount(vt)
                    for vt in config.ALL_VERTEX_TYPES
                    if vt != "Chunk"
                )
            except Exception:
                live = meta["vertex_count"]
            if live > meta["vertex_count"]:
                print(f"Graph grew ({meta['vertex_count']} -> {live}); rebuilding index")
                return build_entity_index(conn, model)
        return meta
    if conn is None:
        from graphrag.schema import connect

        conn = connect()
    return build_entity_index(conn, model, force=force)


def match_entities(mentions: list[dict], index: dict, model=None) -> dict:
    """Match query mentions to graph vertices.

    Args:
        mentions: QueryEntities.entities dicts ({name, type}).
        index: Loaded entity index (load_entity_index output).

    Returns:
        {
          "seeds": [(vtype, vid, matched_name), ...],   # unique, order-stable
          "resolved": [{"mention", "matched", "vtype", "vid", "how"}],  # debug
          "unmatched": [mention, ...],
        }
    """
    embeddings = np.load(_EMBED_FILE)["embeddings"]
    norm_index: dict[str, int] = index["norm_index"]
    vertices: list[dict] = index["vertices"]

    seeds: list[tuple[str, str, str]] = []
    seen_ids: set[tuple[str, str]] = set()
    resolved, unmatched = [], []

    for mention in mentions:
        name = (mention.get("name") or "").strip()
        if not name:
            continue
        norm = normalize_name(name)

        # 1. exact normalized match
        idx = norm_index.get(norm)
        if idx is not None:
            cand = (vertices[idx]["type"], vertices[idx]["id"], vertices[idx]["name"])
            if cand[:2] not in seen_ids:
                seeds.append(cand)
                seen_ids.add(cand[:2])
            resolved.append({"mention": name, "matched": cand[2], "vtype": cand[0], "vid": cand[1], "how": "exact"})
            continue

        q_tokens = set(norm.split())
        q_len = len(q_tokens)
        if q_len > 12:
            unmatched.append(name)
            resolved.append({"mention": name, "matched": None, "vtype": None, "vid": None, "how": "phrase_not_entity"})
            continue

        # 2. embedding similarity, boosted by token overlap (word order and
        #    near-duplicate names make raw cosine unreliable at 180k scale:
        #    "K-2 1000m" vs "K-1 1000m" differ by 0.02 cosine).
        qvec = _embed_texts([norm], model)[0]
        sims = embeddings @ qvec
        cand_idx = np.argsort(-sims)[:25]
        scored = []
        for i in cand_idx:
            cand_tokens = set(normalize_name(vertices[i]["name"]).split())
            overlap = len(q_tokens & cand_tokens) / max(len(cand_tokens), 1)
            # At 180k vertices something ALWAYS clears a bare 0.75 cosine.
            # Accept only: near-identical names, or high cosine + strong
            # token overlap (rejects "men's K-2 1000m canoeing" matching a
            # Women's 50km race-walk athletics event that shares a few tokens).
            if sims[i] >= 0.90 or (overlap >= 0.6 and sims[i] >= 0.80):
                scored.append((0.7 * float(sims[i]) + 0.3 * overlap, i))
        scored.sort(reverse=True)
        # Only the BEST fuzzy candidate: 2nd/3rd near-duplicates (K-1, C-2...)
        # contaminate evidence with wrong-event chunks (doc §15.2). True
        # ambiguity is handled by the evaluator, not by wider retrieval.
        hits = [scored[0][1]] if scored else []
        if hits:
            for i in hits:
                cand = (vertices[i]["type"], vertices[i]["id"], vertices[i]["name"])
                if cand[:2] not in seen_ids:
                    seeds.append(cand)
                    seen_ids.add(cand[:2])
            resolved.append(
                {"mention": name, "matched": vertices[hits[0]]["name"],
                 "vtype": vertices[hits[0]]["type"], "vid": vertices[hits[0]]["id"],
                 "how": f"sim={float(sims[hits[0]]):.2f}x{len(hits)}"}
            )
        else:
            unmatched.append(name)
            resolved.append({"mention": name, "matched": None, "vtype": None, "vid": None, "how": "unmatched"})

    return {"seeds": seeds, "resolved": resolved, "unmatched": unmatched}
