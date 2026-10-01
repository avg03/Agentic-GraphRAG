"""Token + latency collection shared by all pipelines.

Goal (per context.md): every pipeline outputs identical metric families —
latency, tokens, retrieval counts — so results can be compared fairly.

Usage:
    from shared.metrics import Metrics

    m = Metrics()
    with m.timer("retrieval"):
        ...
    m.record_llm(usage_metadata_dict)          # from shared.llm_client
    m.incr("chunks_processed")
    print(m.summary())
"""

import time
from collections import Counter
from contextlib import contextmanager

__all__ = ["Metrics"]


class Metrics:
    """Collects latency, token usage, and arbitrary counters for one run."""

    def __init__(self):
        self.counters: Counter = Counter()
        self.timings: dict[str, list[float]] = {}
        self.llm_tokens: Counter = Counter()  # input/output/total
        self.llm_calls: int = 0
        self.llm_latency_ms_total: float = 0.0
        self._timers: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Counters
    # ------------------------------------------------------------------ #

    def incr(self, key: str, n: int = 1) -> None:
        """Increment a named counter (e.g. chunks_processed, relations_dropped)."""
        self.counters[key] += n

    # ------------------------------------------------------------------ #
    # Timing
    # ------------------------------------------------------------------ #

    @contextmanager
    def timer(self, name: str):
        """Context manager that records elapsed wall-clock ms under `name`."""
        start = time.perf_counter()
        try:
            yield self
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.timings.setdefault(name, []).append(elapsed_ms)

    def record_elapsed(self, name: str, elapsed_ms: float) -> None:
        """Record an externally measured duration (ms)."""
        self.timings.setdefault(name, []).append(elapsed_ms)

    # ------------------------------------------------------------------ #
    # LLM usage
    # ------------------------------------------------------------------ #

    def record_llm(self, usage: dict | None = None, latency_ms: float | None = None) -> None:
        """Record one LLM call.

        Args:
            usage: {"input_tokens", "output_tokens", "total_tokens"} —
                produced by `shared.llm_client` from Gemini usage_metadata.
            latency_ms: Round-trip latency of the call.
        """
        self.llm_calls += 1
        if latency_ms is not None:
            self.llm_latency_ms_total += latency_ms
        if usage:
            self.llm_tokens["input"] += usage.get("input_tokens", 0) or 0
            self.llm_tokens["output"] += usage.get("output_tokens", 0) or 0
            self.llm_tokens["total"] += usage.get("total_tokens", 0) or 0

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        """Flat metric dict — same shape for every pipeline."""

        def _agg(name: str) -> dict:
            values = self.timings.get(name, [])
            if not values:
                return {"count": 0, "total_ms": 0.0}
            return {
                "count": len(values),
                "total_ms": round(sum(values), 2),
                "avg_ms": round(sum(values) / len(values), 2),
            }

        return {
            "counters": dict(self.counters),
            "llm": {
                "calls": self.llm_calls,
                "input_tokens": self.llm_tokens["input"],
                "output_tokens": self.llm_tokens["output"],
                "total_tokens": self.llm_tokens["total"],
                "latency_total_ms": round(self.llm_latency_ms_total, 2),
            },
            "timings": {name: _agg(name) for name in self.timings},
        }

    def pretty(self) -> str:
        """Human-readable one-line-per-group summary."""
        s = self.summary()
        lines = ["== Metrics =="]
        for key, val in sorted(s["counters"].items()):
            lines.append(f"  {key}: {val}")
        llm = s["llm"]
        lines.append(
            "  llm: {calls} calls, {in_tok} in / {out_tok} out / {tot_tok} total tokens, "
            "{lat} ms".format(
                calls=llm["calls"],
                in_tok=llm["input_tokens"],
                out_tok=llm["output_tokens"],
                tot_tok=llm["total_tokens"],
                lat=llm["latency_total_ms"],
            )
        )
        for name, agg in sorted(s["timings"].items()):
            lines.append(f"  {name}: {agg['count']} calls, total {agg['total_ms']} ms")
        return "\n".join(lines)
