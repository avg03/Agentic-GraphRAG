"""Per-run runtime context.

LangGraph state channels only keep declared keys, so non-serializable
objects (Metrics) must travel outside the state — hence this singleton.
"""

__all__ = ["set_metrics", "get_metrics"]

_metrics = None


def set_metrics(metrics) -> None:
    global _metrics
    _metrics = metrics


def get_metrics():
    return _metrics
