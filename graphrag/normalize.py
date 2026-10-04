"""Canonical name normalization shared by resolve, context, and the indexes."""

__all__ = ["normalize_name"]

_STOP_PREFIXES = ("the ", "a ", "an ")
# ASCII + typographic punctuation (en/em dash appear in Wikipedia titles and
# in LLM-generated event names — they must not survive into match keys)
_PUNCT = str.maketrans({c: " " for c in "\"'`.,;:!?()[]{}<>-/&*–—―‒_"})


def normalize_name(name: str) -> str:
    """Canonicalize an entity/event name for dedup keys."""
    lowered = (name or "").strip().lower().translate(_PUNCT)
    lowered = " ".join(lowered.split())
    for prefix in _STOP_PREFIXES:
        if lowered.startswith(prefix) and len(lowered) > len(prefix):
            lowered = lowered[len(prefix) :]
            break
    return lowered
