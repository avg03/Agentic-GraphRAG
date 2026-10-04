"""Standardized result object + metrics contract (shared by all 3 pipelines).

RAGResult lives here so Pipelines 1, 2, and 3 all emit the SAME shape for the
dashboard / comparison tables (architecture doc §18).

The metrics contract (validate_metrics) defines the core metrics every
pipeline MUST emit; pipeline-specific extras are checked only for their
pipeline (graph_rag / agentic_graphrag). v2 additions (architecture v2 §18)
are additive: pipelines that don't emit them yet simply report missing.
"""

from dataclasses import dataclass, field, asdict

__all__ = [
    "RAGResult", "validate_metrics", "METRICS_CONTRACT",
    "AGENTIC_EXTRAS", "GRAPHRAG_EXTRAS",
]

_CORE = [
    # token efficiency
    "context_tokens", "input_tokens", "output_tokens", "total_tokens",
    # latency
    "latency_ms", "retrieval_latency_ms", "llm_latency_ms",
    # retrieval
    "n_chunks", "n_citations",
    # quality flags
    "evidence_sufficient", "confidence",
]

AGENTIC_EXTRAS = [
    # v1
    "steps", "tools_used", "strategy_changed", "stop_reason",
    "reason_codes", "time_per_tool", "tokens_per_op",
    # v2 (architecture v2 §18)
    "contexts_investigated", "vector_fallback_count",
    "reformulation_count", "new_evidence_per_step",
]

GRAPHRAG_EXTRAS = [
    # v2 (architecture v2 §18)
    "unique_seed_vertices", "unique_context_roots", "unique_action_events",
    "unique_chunks", "scope_coverage", "retrieval_mode",
    "duplicate_chunk_count",
]


@dataclass
class RAGResult:
    """Standardized comparable result object for every pipeline/question."""

    pipeline: str
    question: str
    answer: str
    citations: list
    retrieved_chunks: list
    context_tokens: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    evidence_sufficient: bool | None = None
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def metrics_view(self) -> dict:
        """Flat dict for validate_metrics / dashboard rows."""
        view = {
            "context_tokens": self.context_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "n_chunks": len(self.retrieved_chunks),
            "n_citations": len(self.citations),
            "evidence_sufficient": self.evidence_sufficient,
            "confidence": self.confidence,
        }
        for k in (*AGENTIC_EXTRAS, *GRAPHRAG_EXTRAS):
            if k in self.metadata:
                view[k] = self.metadata[k]
        return view


METRICS_CONTRACT = {
    "core": _CORE,
    "agentic_extras": AGENTIC_EXTRAS,
    "graphrag_extras": GRAPHRAG_EXTRAS,
}


def validate_metrics(metrics_view: dict, pipeline: str) -> dict:
    """Check a pipeline's per-question metrics against the contract.

    Returns {"missing_core": [...], "missing_extras": [...], "ok": bool}.
    Extra metrics are fine; the core set + the pipeline's own extras must be
    present.
    """
    missing_core = [k for k in _CORE if metrics_view.get(k) is None]
    extras: list[str] = []
    if pipeline == "agentic_graphrag":
        extras = AGENTIC_EXTRAS
    elif pipeline == "graph_rag":
        extras = GRAPHRAG_EXTRAS
    missing_extras = [k for k in extras if metrics_view.get(k) is None]
    return {
        "missing_core": missing_core,
        "missing_extras": missing_extras,
        "ok": not missing_core and not missing_extras,
    }
