"""Deterministic context/hierarchy derivation (architecture v2 §3.2, §5.2).

Turns an Olympics event name or Wikipedia doc title into structured,
queryable context metadata:

    {
      "root_context":  "2012 Summer Olympics",          # top-level scope
      "context_path":  ["2012 Summer Olympics",
                        "Canoeing",
                        "Men's K-2 1000 metres"],
      "context_keys":  ["olympics:2012",
                        "sport:canoeing",
                        "event:mens_k2_1000_metres"],
      "context_level": "event"        # edition < sport < event
    }

Zero LLM calls: the corpus is uniformly Wikipedia pages, and 2210/2951
titles follow "Sport at the YYYY Season Olympics – Event". Anything else
falls back to a self-rooted edition-level context (keys: topic:/year:),
so every ActionEvent still gets a stable scope.

Context keys are the retrieval primitive (v2 §2: a-b-c lineage strings are
display-only); the display path is provenance for prompts and debugging.
"""

import json
import re

from graphrag.normalize import normalize_name

__all__ = [
    "derive_context",
    "merge_contexts",
    "context_to_attrs",
    "context_keys_from_attrs",
    "sports_vocab_from_titles",
]

# edition: "1994 Winter Olympics" / "2008 Summer Olympics"
_EDITION_RE = re.compile(r"\b(\d{4})\s+(Summer|Winter)\s+Olympics\b", re.IGNORECASE)
# canonical Wikipedia title: "Canoeing at the 2012 Summer Olympics – Men's K-2 1000 metres"
_CANONICAL_RE = re.compile(
    r"^(?P<sport>.+?)\s+at\s+the\s+(?P<year>\d{4})\s+(?P<season>Summer|Winter)\s+Olympics"
    r"(?:\s*[–-]\s*(?P<event>.+))?$",
    re.IGNORECASE,
)
# round qualifiers that sub-divide one event (stripped from event descriptors
# so heats/semifinals/finals resolve to the same sibling group)
_ROUND_WORDS = r"(final\s+[ab]\b|finals?|semi[\s-]*finals?|quarter[\s-]*finals?|heats?|repechage|qualifications?|heat\s+\d+|round\s+\d+)"
_ROUND_PREFIX_RE = re.compile(rf"^(?:the\s+)?{_ROUND_WORDS}\s*(?:of|for|at|in)\s+", re.IGNORECASE)
_ROUND_SUFFIX_RE = re.compile(rf"[\s,]+(?:{_ROUND_WORDS})\s*$", re.IGNORECASE)
_BOILERPLATE_SUFFIX = re.compile(r"[\s,]+\b(event|competition|race|contest)\s*$", re.IGNORECASE)
_CONNECTOR_TAIL = re.compile(r"[\s,]+(?:(?:at|in|of|for|during|the)\s*)+$", re.IGNORECASE)
_CONNECTOR_HEAD = re.compile(r"^(?:(?:at|in|of|for|during|the)\s+)+", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
# the canonical split assumes the text before "at the" is a SPORT; event-first
# names ("Men's 1000 metres ... at the 2006 Winter Olympics") violate that —
# detect them by event-descriptor markers instead of relying on the vocab
_EVENTISH_SLOT_RE = re.compile(
    r"\d|metre|meter|\bkg\b|\bmen\b|\bwomen\b|heat|final|semi|quarter|repechage|race|competition",
    re.IGNORECASE,
)


def _norm_key(value: str) -> str:
    """Normalized key value: canonical name, underscores (v2 §5.2 style)."""
    return normalize_name(value).replace(" ", "_")


def _strip_rounds(text: str) -> str:
    cleaned = _ROUND_PREFIX_RE.sub("", text or "")
    cleaned = _ROUND_SUFFIX_RE.sub("", cleaned)
    cleaned = _BOILERPLATE_SUFFIX.sub("", cleaned)
    return cleaned.strip(" ,–-")


def _extract_sport(text: str, sports_vocab: set[str] | None) -> str:
    """Find a known Olympic sport mentioned in the text (longest match wins)."""
    if not sports_vocab:
        return ""
    norm_text = f" {normalize_name(text)} "
    best = ""
    for sport in sports_vocab:
        norm_sport = normalize_name(sport)
        if norm_sport and f" {norm_sport} " in norm_text and len(norm_sport) > len(best):
            best = sport
    return best


def _flexible_context(raw: str, before: str, after: str, sports_vocab: set[str] | None) -> tuple[str, str]:
    """(sport, event) from a non-canonical name. `before`/`after` are the
    texts around the edition mention; word-level cleanup only."""
    before = _CONNECTOR_TAIL.sub("", before or "").strip(" ,–-")
    after = _CONNECTOR_HEAD.sub("", after or "").strip(" ,–-")
    after = _strip_rounds(after)

    event = after or _strip_rounds(before)
    sport = _extract_sport(raw, sports_vocab)
    if sport and event:
        ev_norm = normalize_name(event)
        sport_norm = normalize_name(sport)
        if ev_norm == sport_norm:
            event = ""
        elif f" {sport_norm} " in f" {ev_norm} ":
            event = _strip_rounds(re.sub(re.escape(sport), "", event, count=1, flags=re.IGNORECASE))
    return sport, event


def derive_context(name: str, sports_vocab: set[str] | None = None) -> dict | None:
    """Derive structured context from an event name or doc title.

    Returns None only for empty input. Never raises.
    """
    raw = (name or "").strip()
    if not raw:
        return None

    edition_m = _EDITION_RE.search(raw)
    if edition_m:
        year = edition_m.group(1)
        season = edition_m.group(2).capitalize()
        root = f"{year} {season} Olympics"
        keys = [f"olympics:{year}"]
        path = [root]

        canonical = _CANONICAL_RE.match(raw)
        if canonical and _EVENTISH_SLOT_RE.search(canonical.group("sport")):
            canonical = None  # sport slot is really an event descriptor
        if canonical and sports_vocab and normalize_name(canonical.group("sport")) not in {
            normalize_name(s) for s in sports_vocab
        }:
            # not a known sport AND no event markers: still accept the split
            # (titles are reliable); only bail when the slot is multi-word
            # and unconstrained ("Race schedule at 2008 Summer Olympics ...")
            if len(canonical.group("sport").split()) > 2:
                canonical = None
        if canonical:
            sport = canonical.group("sport").strip()
            event = (canonical.group("event") or "").strip()
        else:
            parts = _EDITION_RE.split(raw, maxsplit=1)
            # split with 2 capture groups -> [before, year, season, tail]
            before = parts[0] if parts else raw
            tail = parts[3] if len(parts) > 3 else ""
            sport, event = _flexible_context(raw, before, tail, sports_vocab)

        if sport:
            keys.append(f"sport:{_norm_key(sport)}")
            path.append(sport)
        if event:
            event = _strip_rounds(event)
            if event and normalize_name(event) not in (
                normalize_name(sport or ""), normalize_name(root)
            ):
                keys.append(f"event:{_norm_key(event)}")
                path.append(event)

        if any(k.startswith("event:") for k in keys):
            level = "event"
        elif any(k.startswith("sport:") for k in keys):
            level = "sport"
        else:
            level = "edition"
        return {
            "root_context": root,
            "context_path": path,
            "context_keys": keys,
            "context_level": level,
        }

    # ---- non-Olympics fallback: self-rooted, keyed by topic (+ year) ----
    keys = [f"topic:{_norm_key(raw)}"]
    year_m = _YEAR_RE.search(raw)
    if year_m:
        keys.append(f"year:{year_m.group(1)}")
    return {
        "root_context": raw,
        "context_path": [raw],
        "context_keys": keys,
        "context_level": "edition",
    }


def merge_contexts(event_ctx: dict | None, title_ctx: dict | None) -> dict | None:
    """Prefer event-derived context; fill gaps from the doc-title context.

    Used when an extracted ActionEvent name is messier than its source doc
    title: the title usually carries the clean sport/edition split. The
    edition root stays the event's own; keys union; display path unions.
    """
    if event_ctx is None:
        return title_ctx
    if title_ctx is None:
        return event_ctx

    merged_keys = list(event_ctx["context_keys"])
    for key in title_ctx["context_keys"]:
        if key not in merged_keys:
            merged_keys.append(key)

    path = list(event_ctx["context_path"])
    seen = {normalize_name(p) for p in path}
    for node in title_ctx["context_path"]:
        if normalize_name(node) not in seen:
            path.append(node)
            seen.add(normalize_name(node))

    # root_context prefers the clean edition from the title when the event
    # name only implied one (e.g. "Final A at 1994 Winter Olympics")
    root = event_ctx["root_context"]
    if title_ctx["context_level"] in ("sport", "event"):
        root = title_ctx["root_context"]
    level = event_ctx["context_level"]
    if level == "edition" and title_ctx["context_level"] != "edition":
        level = "sport"
    return {
        "root_context": root,
        "context_path": path,
        "context_keys": merged_keys,
        "context_level": level,
    }


def sports_vocab_from_titles(titles: dict[str, str]) -> set[str]:
    """Collect sport names from canonical corpus titles
    ("Canoeing at the 2012 Summer Olympics – ...") — used to validate the
    sport slot and tag sports in non-canonical event names."""
    vocab = set()
    for title in titles.values():
        m = _CANONICAL_RE.match(title or "")
        if m and m.group("sport"):
            vocab.add(m.group("sport").strip())
    return vocab


def context_to_attrs(context: dict | None) -> dict[str, str]:
    """Encode a context dict as string vertex attributes (TigerGraph wire
    format is string-typed; lists are JSON-encoded)."""
    if not context:
        return {}
    return {
        "root_context": context["root_context"],
        "context_level": context["context_level"],
        "context_path": json.dumps(context["context_path"], ensure_ascii=False),
        "context_keys": json.dumps(context["context_keys"], ensure_ascii=False),
    }


def context_keys_from_attrs(attrs: dict) -> list[str]:
    """Read context_keys back out of vertex attributes (defensive: accepts
    both the JSON string form and a raw list)."""
    raw = attrs.get("context_keys") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = [raw]
    return [k for k in raw if isinstance(k, str) and k]
