"""Scope resolution: candidate context keys -> ranked roots (v2 §3.3/§3.4/§6.3).

Separates DISCOVERY from TRAVERSAL: this module decides which context roots
are worth investigating and in what order; the traversal layer (traverse.py)
then walks relationships inside the chosen scope.

Specificity-first (v2 §6.3): exact event > sport > edition. Broad roots are
only expanded when the current scope fails the coverage gate — the ladder
helper exposes that order.
"""

from graphrag import config

__all__ = [
    "key_level",
    "key_specificity",
    "resolve_scope",
    "expansion_ladder",
    "dedupe_event_ids",
]

# key prefix -> (level name, specificity rank; higher = more specific)
_KEY_LEVELS = {
    "event": ("event", 4),
    "discipline": ("event", 4),  # v2 §5.2 treats disciplines as event-level groups
    "sport": ("sport", 3),
    "olympics": ("edition", 2),
    "year": ("edition", 1),
    "topic": ("edition", 1),
}

# the meaningful expansion hierarchy (v2 §6.3): a failing scope broadens
# along this ladder and STOPS at the edition level — topic/year keys are
# discovery helpers, never expansion targets
_HIERARCHY = ["event", "sport", "olympics"]


def key_level(key: str) -> str:
    """Level name for a context key ('event' | 'sport' | 'edition')."""
    prefix = key.split(":", 1)[0]
    return _KEY_LEVELS.get(prefix, ("edition", 1))[0]


def key_specificity(key: str) -> int:
    """Specificity rank for a context key (4 exact event ... 1 bare topic)."""
    prefix = key.split(":", 1)[0]
    return _KEY_LEVELS.get(prefix, ("edition", 1))[1]


def resolve_scope(
    matches: list[dict],
    max_roots: int | None = None,
    min_events: int = 1,
) -> list[dict]:
    """Rank matched context keys into investigation roots.

    Args:
        matches: match_contexts() "matches" entries
            ({phrase, key, display, how, vertex_ids}).
        max_roots: Cap on returned roots (config.SCOPE_MAX_CONTEXT_ROOTS).
        min_events: Drop roots with fewer resolvable events (noise guard).

    Returns:
        Ordered roots, most specific first:
            [{"key", "display", "level", "specificity", "vertex_ids"}...]
    """
    if max_roots is None:
        max_roots = config.SCOPE_MAX_CONTEXT_ROOTS

    roots: dict[str, dict] = {}
    for m in matches:
        key = m["key"]
        vids = [v for v in m.get("vertex_ids", []) if v]
        if key in roots:
            existing = roots[key]["vertex_ids"]
            roots[key]["vertex_ids"] = sorted(set(existing) | set(vids))
            continue
        roots[key] = {
            "key": key,
            "display": m.get("display", key),
            "level": key_level(key),
            "specificity": key_specificity(key),
            "vertex_ids": sorted(set(vids)),
        }

    ranked = [r for r in roots.values() if len(r["vertex_ids"]) >= min_events]
    # most specific first; ties broken by evidence volume (bigger scope first
    # among equal specificity), then key string for determinism
    ranked.sort(key=lambda r: (-r["specificity"], -len(r["vertex_ids"]), r["key"]))
    return ranked[:max_roots]


def expansion_ladder(current_keys: list[str]) -> list[str]:
    """Next-broader key prefixes for insufficient scopes (v2 §6.3 ladder).

    event -> sport -> olympics, stopping at the edition level. Returns the
    hierarchy prefixes broader than the most specific key currently
    investigated (immediate next step first); [] when already at edition
    breadth — the gate should fall back to vector search instead.
    """
    if not current_keys:
        return list(reversed(_HIERARCHY[1:]))
    broadest = min(key_specificity(k) for k in current_keys)
    broader = [p for p in _HIERARCHY if key_specificity(p) < broadest]
    # immediate next step first (sport before olympics when broadening an event)
    broader.sort(key=key_specificity, reverse=True)
    return broader


def dedupe_event_ids(event_ids: list[str]) -> list[str]:
    """Stable-order dedup of candidate events across discovery paths
    (v2 §15.4: the same event may surface from a sport root, an edition
    root, and an entity seed)."""
    seen: set[str] = set()
    out: list[str] = []
    for eid in event_ids:
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out
