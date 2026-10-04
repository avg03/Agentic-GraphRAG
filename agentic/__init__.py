"""Agentic GraphRAG (Pipeline 3) — investigation layer over Pipelines 1 & 2."""

from .config import (
    MAX_STEPS,
    MAX_GRAPH_CALLS,
    MAX_SCOPE_CALLS,
    MAX_VECTOR_CALLS,
    MAX_COMMUNITY_CALLS,
    MAX_CONTEXT_ROOTS,
)

__all__ = [
    "MAX_STEPS",
    "MAX_GRAPH_CALLS",
    "MAX_SCOPE_CALLS",
    "MAX_VECTOR_CALLS",
    "MAX_COMMUNITY_CALLS",
    "MAX_CONTEXT_ROOTS",
]
