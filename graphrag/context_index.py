"""Context index: context key / phrase -> graph ActionEvent vertices (v2 §5).

The complement of the entity index (mention -> vertex): this maps
higher-level scopes to the events inside them,

    "olympics:2008"   -> [ActionEvent, ActionEvent, ...]
    "sport:canoeing"  -> [ActionEvent, ...]

giving sibling discovery a deterministic candidate layer. Graph
relationships still live in TigerGraph; this index only discovers WHICH
events belong to a requested scope.

Build is fully offline: resolve is re-run from the cached extraction JSONL
plus doc titles from Chroma (zero LLM calls), and the index is cached to
disk like the entity index:

    graphrag/context_index.json   key -> {display, count, vertex_ids}
    graphrag/context_index.npz    BGE embedding per key (phrase matching)

Edition keys get season-qualified subkeys ("olympics:1992_winter") when a
year hosts both seasons, so phrase queries like "1992 Winter Olympics"
resolve precisely; the bare key ("olympics:1992") stays the union.

Matching a phrase:
    1. exact normalized key match ("olympics:2008")
    2. otherwise embedding similarity against key displays ("2008 Summer
       Olympics"), with the same strict acceptance rule as entity matching
    3. deterministic ranking, best candidate only

Usage:
    python -m graphrag.context_index --build
    python -m graphrag.context_index --build --probe "1992 Winter Olympics" canoeing
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from graphrag.context import context_keys_from_attrs
from graphrag.extract import load_cached_extractions
from graphrag.ingest_pipeline import load_chunks_from_chroma
from graphrag.resolve import resolve_extractions
from shared.llm_client import get_client  # noqa: F401  (ensures env loaded)

__all__ = [
    "build_context_index",
    "load_context_index",
    "match_contexts",
]

from graphrag.entity_index import _embed_texts, fuzzy_accept

_INDEX_DIR = Path(__file__).resolve().parent
_CATALOG_FILE = _INDEX_DIR / "context_index.json"
_EMBED_FILE = _INDEX_DIR / "context_index.npz"


def _load_titles(collection=None) -> dict[str, str]:
    """{doc_id: title} from Chroma metadata (titles anchor event context)."""
    if collection is None:
        from database.chroma_client import get_collection

        collection = get_collection()
    titles: dict[str, str] = {}
    offset, batch = 0, 5000
    while True:
        stored = collection.get(include=["metadatas"], limit=batch, offset=offset)
        metas = stored.get("metadatas") or []
        if not metas:
            break
        for meta in metas:
            doc_id = (meta or {}).get("doc_id", "")
            title = (meta or {}).get("title", "")
            if doc_id and title and doc_id not in titles:
                titles[doc_id] = title
        if len(metas) < batch:
            break
        offset += batch
    return titles


def _key_display(key: str) -> str:
    """Fallback human phrase for a key: 'sport:canoeing' -> 'canoeing'."""
    return key.split(":", 1)[1].replace("_", " ") if ":" in key else key


def _season_of(root: str) -> str:
    """"1992 Winter Olympics" -> "winter" ("" when no season word)."""
    lowered = (root or "").lower()
    for season in ("summer", "winter"):
        if season in lowered:
            return season
    return ""


def build_context_index(collection=None, model=None) -> dict:
    """(Re)build the context index from cached extractions + Chroma titles."""
    records = load_cached_extractions()
    if not records:
        raise RuntimeError("No cached extractions — run graphrag.ingest_pipeline first.")
    titles = _load_titles(collection)

    resolved = resolve_extractions(records, titles=titles)
    events = resolved["vertices"].get("ActionEvent", {})

    # key -> {(vid, root_context)} members + display preference votes
    members: dict[str, set] = defaultdict(set)
    roots: dict[str, Counter] = defaultdict(Counter)
    for vid, data in events.items():
        attrs = data.get("attributes", {})
        root = attrs.get("root_context", "")
        for key in context_keys_from_attrs(attrs):
            members[key].add((vid, root))
            if key.startswith("olympics:"):
                roots[key][root] += 1

    keys: dict[str, dict] = {}
    for key, mem in members.items():
        entry = {
            "display": _key_display(key),
            "vertex_ids": sorted({vid for vid, _ in mem}),
        }
        if key.startswith("olympics:"):
            year = key.split(":", 1)[1]
            entry["display"] = f"{year} olympics"
            # season-qualified subkeys when both seasons share the year
            seasons = {_season_of(root): root for root in roots[key]}
            seasons.pop("", None)
            if len(seasons) > 1:
                for season, root in sorted(seasons.items()):
                    sub = f"olympics:{year}_{season}"
                    keys[sub] = {
                        "display": root.lower(),
                        "vertex_ids": sorted({vid for vid, r in mem if r == root}),
                    }
        keys[key] = entry

    key_list = sorted(keys)
    phrases = [keys[k]["display"] for k in key_list]
    embeddings = _embed_texts(phrases, model)
    np.savez_compressed(_EMBED_FILE, embeddings=embeddings)

    meta = {
        "vertex_count": len(events),
        "keys": {k: keys[k] for k in key_list},
        "phrases": phrases,
        "display_to_key": {keys[k]["display"]: k for k in key_list},
    }
    _CATALOG_FILE.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(
        f"Context index built: {len(key_list)} keys over {len(events)} ActionEvents"
        f" -> {_CATALOG_FILE.name}"
    )
    return meta


def load_context_index(force: bool = False) -> dict:
    """Load the cached context index (offline; rebuilds only with force)."""
    if force or not _CATALOG_FILE.exists() or not _EMBED_FILE.exists():
        return build_context_index()
    meta = json.loads(_CATALOG_FILE.read_text(encoding="utf-8"))
    meta.setdefault("display_to_key", {e["display"]: k for k, e in meta["keys"].items()})
    return meta


def _exact_key(phrase: str, keys: dict) -> str | None:
    """Direct key hit: already normalized, or trivially prefixable."""
    p = phrase.strip().lower().replace(" ", "_")
    if p in keys:
        return p
    # "olympics 2008" / "sport canoeing" style without the colon
    if ":" not in p:
        for prefix in ("olympics", "sport", "event", "year", "topic", "discipline"):
            if p.startswith(prefix + "_"):
                cand = f"{prefix}:{p[len(prefix) + 1 :]}"
                if cand in keys:
                    return cand
    return None


def match_contexts(phrases: list[str], index: dict, model=None) -> dict:
    """Match query phrases to context keys -> candidate ActionEvent vertices.

    Args:
        phrases: Candidate context keys or scope phrases ("olympics:2008",
            "2008 Summer Olympics", "canoeing").
        index: load_context_index() output.

    Returns:
        {"matches": [{"phrase", "key", "display", "how", "vertex_ids"}...],
         "unmatched": [phrase, ...]}

    vertex_ids are ActionEvent ids (single vertex type, so bare ids).
    """
    keys = index["keys"]
    phrases_list: list[str] = index["phrases"]
    display_to_key = index["display_to_key"]
    embeddings = np.load(_EMBED_FILE)["embeddings"]

    matches, unmatched = [], []
    for phrase in phrases:
        text = (phrase or "").strip()
        if not text:
            continue

        key = _exact_key(text, keys)
        how = "exact"
        if key is None:
            # semantic: phrase -> nearest key display, best candidate only
            from graphrag.normalize import normalize_name

            qvec = _embed_texts([normalize_name(text)], model)[0]
            sims = embeddings @ qvec
            cand_idx = np.argsort(-sims)[:5]
            q_tokens = set(normalize_name(text).split())
            scored = []
            for i in cand_idx:
                disp_tokens = set(normalize_name(phrases_list[i]).split())
                overlap = len(q_tokens & disp_tokens) / max(len(disp_tokens), 1)
                if fuzzy_accept(float(sims[i]), overlap) or float(sims[i]) >= 0.85:
                    scored.append((0.7 * float(sims[i]) + 0.3 * overlap, i))
            if scored:
                scored.sort(reverse=True)
                best = scored[0][1]
                key = display_to_key.get(phrases_list[best])
                how = f"sim={float(sims[best]):.2f}"
        if key:
            matches.append({
                "phrase": text,
                "key": key,
                "display": keys[key]["display"],
                "how": how,
                "vertex_ids": keys[key]["vertex_ids"],
            })
        else:
            unmatched.append(text)

    return {"matches": matches, "unmatched": unmatched}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="(Re)build the context index")
    parser.add_argument("--probe", nargs="+", default=None, help="Test-match phrases")
    args = parser.parse_args()

    if args.build or not _CATALOG_FILE.exists():
        build_context_index()
    if args.probe:
        index = load_context_index()
        result = match_contexts(args.probe, index)
        for m in result["matches"]:
            print(f"  {m['phrase']!r} -> {m['key']} ({m['how']}) {len(m['vertex_ids'])} events")
        for u in result["unmatched"]:
            print(f"  {u!r} -> UNMATCHED")


if __name__ == "__main__":
    main()
