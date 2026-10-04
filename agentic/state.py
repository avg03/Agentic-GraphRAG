"""AgentState (LangGraph state) + trace + standardized RAGResult (doc §13/§16/§18)."""

import time
from typing import Annotated, Any, TypedDict

from operator import add

from shared.result import RAGResult  # single definition, shared by all pipelines

__all__ = ["AgentState", "RAGResult", "new_trace_step"]


class AgentState(TypedDict, total=False):
    """Shared investigation state flowing through the LangGraph nodes."""

    # input
    question: str

    # plan (v2: planner also classifies the question + proposes scope)
    plan: dict                      # PlanOutput dict
    slots: list                     # answer_slots (list of dicts)
    question_type: str              # local_fact | multi_hop | collection_aggregation | overview

    # scope/context (v2 §11)
    candidate_contexts: list        # [{key, display, level, n_events}] ranked roots
    investigated_contexts: list     # context keys already explored (dedupe guard)
    analysis: dict                  # QueryAnalysis dict reused by graph tools

    # supervision
    next_action: str                # GRAPH_SEARCH | GRAPH_SCOPE | VECTOR_SEARCH | COMMUNITY_SEARCH | REFORMULATE | STOP
    target_slot: str
    target_context: str             # context key the scope action investigates
    subquery: str                   # current (possibly reformulated) query
    reason_code: str

    # evidence
    graph_paths: Annotated[list, add]      # relationship path strings (append-only)
    graph_relationships: Annotated[list, add]
    chunks: Annotated[list, add]           # chunk dicts w/ source tags (append-only; deduped at the end)
    unmatched_mentions: Annotated[list, add]
    slot_statuses: dict             # slot_id -> latest SlotStatus dict
    evidence_round: int
    evidence_stats: dict            # total_chunks, distinct_events, new_chunks_last_step, marginal_gain
    scope_coverage: str             # latest coverage verdict from the evaluator

    # bookkeeping
    trace: Annotated[list, add]     # step dicts
    step_count: int
    graph_calls: int
    scope_calls: int
    vector_calls: int
    community_calls: int
    seen_queries: list              # (action, subquery) pairs already executed
    stop_reason: str
    llm_usage: dict                 # aggregated token counts

    # outputs (answer node)
    context_tokens: int
    final_answer: dict
    final_chunks: list
    duplicate_chunks_dropped: int
    last_tool_result: str


def new_trace_step(node: str, **fields) -> dict:
    """One trace entry (doc §16): what the system did, no hidden reasoning."""
    step = {"ts": round(time.time(), 3), "node": node}
    step.update(fields)
    return step
