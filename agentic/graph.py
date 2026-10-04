"""LangGraph wiring (v2 §10.2/§13): planner → context resolver → supervisor →
tools (graph / scope / vector / community) → evaluator → … → answer."""

from langgraph.graph import StateGraph, END

from agentic import config
from agentic.state import AgentState, new_trace_step
from agentic._runtime import get_metrics
from agentic.planner import plan_question
from agentic.context_resolver import resolve_candidate_contexts
from agentic.supervisor import decide_next_action
from agentic.evaluator import evaluate_evidence
from agentic.context_builder import build_context
from agentic.answer import generate_answer

__all__ = [
    "build_agent_graph",
    "GRAPH_SEARCH",
    "GRAPH_SCOPE",
    "VECTOR_SEARCH",
    "COMMUNITY_SEARCH",
    "REFORMULATE",
    "STOP",
]

GRAPH_SEARCH = "GRAPH_SEARCH"
GRAPH_SCOPE = "GRAPH_SCOPE"
VECTOR_SEARCH = "VECTOR_SEARCH"
COMMUNITY_SEARCH = "COMMUNITY_SEARCH"
REFORMULATE = "REFORMULATE"
STOP = "STOP"


# --------------------------------------------------------------------------- #
# Node wrappers (each updates the shared AgentState)
# --------------------------------------------------------------------------- #

def planner_node(state: AgentState) -> dict:
    import time

    t0 = time.perf_counter()
    plan, _ = plan_question(state["question"], metrics=get_metrics())
    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "plan": plan.model_dump(),
        "slots": [s.model_dump() for s in plan.answer_slots],
        "question_type": plan.question_type,
        "trace": [new_trace_step(
            "planner", action="decompose",
            reason="multiple answer slots" if len(plan.answer_slots) > 1 else "single slot",
            question_type=plan.question_type, scope_root=plan.scope_root,
            slots=[s.id for s in plan.answer_slots], latency_ms=round(elapsed, 1),
        )],
    }


def context_resolver_node(state: AgentState) -> dict:
    """v2 §10.2: candidate context keys -> ranked roots (deterministic)."""
    import time

    t0 = time.perf_counter()
    plan = state.get("plan", {})
    roots, matches = resolve_candidate_contexts(
        plan.get("candidate_context_keys", []),
        scope_root=plan.get("scope_root", ""),
        max_roots=config.MAX_CONTEXT_ROOTS,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    candidates = [
        {
            "key": r["key"],
            "display": r["display"],
            "level": r["level"],
            "n_events": len(r["vertex_ids"]),
        }
        for r in roots
    ]
    return {
        "candidate_contexts": candidates,
        "investigated_contexts": [],
        "trace": [new_trace_step(
            "context_resolver", action="resolve_scope",
            reason="rank candidate context roots",
            candidates=[c["key"] for c in candidates],
            latency_ms=round(elapsed, 1),
        )],
    }


def supervisor_node(state: AgentState) -> dict:
    import time

    t0 = time.perf_counter()
    metrics = get_metrics()
    budget = {
        "steps": state.get("step_count", 0),
        "max_steps": config.MAX_STEPS,
        "graph_calls": state.get("graph_calls", 0),
        "max_graph": config.MAX_GRAPH_CALLS,
        "scope_calls": state.get("scope_calls", 0),
        "max_scope": config.MAX_SCOPE_CALLS,
        "vector_calls": state.get("vector_calls", 0),
        "max_vector": config.MAX_VECTOR_CALLS,
        "community_calls": state.get("community_calls", 0),
        "max_community": config.MAX_COMMUNITY_CALLS,
    }
    decision, _ = decide_next_action(
        state["question"], state.get("slots", []),
        state.get("slot_statuses", {}), state.get("trace", []),
        budget, metrics=metrics,
        question_type=state.get("question_type", ""),
        candidate_contexts=state.get("candidate_contexts", []),
        investigated_contexts=state.get("investigated_contexts", []),
        evidence_stats=state.get("evidence_stats", {}),
    )
    elapsed = (time.perf_counter() - t0) * 1000
    update = {
        "next_action": decision.action,
        "target_slot": decision.target_slot,
        "target_context": decision.target_context,
        "reason_code": decision.reason_code,
        "trace": [new_trace_step(
            "supervisor", action=decision.action,
            reason_code=decision.reason_code, target=decision.target_slot,
            target_context=decision.target_context,
            note=decision.note, latency_ms=round(elapsed, 1),
        )],
    }
    if decision.subquery:
        update["subquery"] = decision.subquery
    elif decision.action == REFORMULATE and decision.target_slot:
        # reformulation must rewrite the subquery; fall back to slot question
        for slot in state.get("slots", []):
            if slot["id"] == decision.target_slot:
                update["subquery"] = slot["question"]
                break
    return update


def _chunks_stats(all_chunks: list) -> dict:
    """Bookkeeping for coverage/marginal-gain prompts (v2 §9)."""
    event_ids = {
        eid for c in all_chunks for eid in (c.get("event_ids") or [])
    }
    return {"total_chunks": len(all_chunks), "distinct_events": len(event_ids)}


def _tool_node(tool_name: str):
    def node(state: AgentState) -> dict:
        import time

        t0 = time.perf_counter()
        subquery = state.get("subquery") or state["question"]
        metrics = get_metrics()
        prev_ids = {c.get("id") for c in state.get("chunks", [])}
        if tool_name == GRAPH_SEARCH:
            from agentic.tools import graph_search

            result = graph_search(subquery, max_paths=config.MAX_PATHS)
            new_chunks = result.get("chunks", [])
            update = {
                "graph_paths": result.get("paths", []),
                "unmatched_mentions": result.get("unmatched", []),
                "chunks": new_chunks,
                "graph_calls": state.get("graph_calls", 0) + 1,
            }
            outcome = (
                f"{len(new_chunks)} chunks ({result.get('retrieval_mode', 'local')} mode), "
                f"entities={result.get('entities', [])}"
            )
        elif tool_name == GRAPH_SCOPE:
            from agentic.tools import scope_search

            target = state.get("target_context", "")
            result = scope_search(
                subquery,
                context_keys=[target] if target else None,
                max_paths=config.MAX_PATHS,
            )
            new_chunks = result.get("chunks", [])
            investigated = list(state.get("investigated_contexts", []))
            if target:
                investigated.append(target)
            elif result.get("scope_roots"):
                investigated.extend(r["key"] for r in result["scope_roots"])
            update = {
                "graph_paths": result.get("paths", []),
                "unmatched_mentions": result.get("unmatched", []),
                "chunks": new_chunks,
                "scope_calls": state.get("scope_calls", 0) + 1,
                "investigated_contexts": investigated,
            }
            roots = ", ".join(r.get("key", "") for r in result.get("scope_roots", []))
            outcome = (
                f"{len(new_chunks)} chunks from sibling events (roots: {roots or 'none'})"
            )
        elif tool_name == VECTOR_SEARCH:
            from agentic.tools import vector_search

            result = vector_search(subquery)
            new_chunks = result.get("chunks", [])
            update = {"chunks": new_chunks, "vector_calls": state.get("vector_calls", 0) + 1}
            outcome = f"{len(new_chunks)} chunks"
        else:
            from agentic.tools import community_search

            result = community_search(subquery)
            new_chunks = []
            update = {"community_calls": state.get("community_calls", 0) + 1}
            outcome = f"{len(result.get('communities', []))} doc communities"
        elapsed = (time.perf_counter() - t0) * 1000

        # marginal-gain bookkeeping (v2 §9/§19): new evidence this step
        new_count = sum(1 for c in new_chunks if c.get("id") not in prev_ids)
        all_chunks = state.get("chunks", []) + new_chunks
        stats = _chunks_stats(all_chunks)
        stats["new_chunks_last_step"] = new_count
        stats["marginal_gain"] = (
            "increasing" if new_count >= 3 else "flat" if new_count >= 1 else "diminishing"
        )

        trace = [new_trace_step(
            "retrieval", step=state.get("step_count", 0) + 1,
            tool=tool_name, target_slot=state.get("target_slot", ""),
            target_context=state.get("target_context", ""),
            subquery=subquery, result=outcome,
            new_chunks=new_count, latency_ms=round(elapsed, 1),
        )]
        update["trace"] = trace
        update["step_count"] = state.get("step_count", 0) + 1
        update["seen_queries"] = state.get("seen_queries", []) + [(tool_name, subquery)]
        update["last_tool_result"] = outcome
        update["evidence_stats"] = stats
        if metrics is not None:
            metrics.record_elapsed(f"tool_{tool_name.lower()}", elapsed)
            metrics.incr(f"chunks_{tool_name.lower()}", len(new_chunks))
            metrics.incr("new_evidence_chunks_total", new_count)
        return update

    return node


def evaluator_node(state: AgentState) -> dict:
    import time

    t0 = time.perf_counter()
    metrics = get_metrics()
    evaluation, _ = evaluate_evidence(
        state["question"], state.get("slots", []),
        state.get("graph_paths", []), state.get("chunks", []),
        state.get("unmatched_mentions", []), metrics=metrics,
        question_type=state.get("question_type", ""),
        scope_roots=state.get("candidate_contexts", []),
        evidence_stats=state.get("evidence_stats", {}),
    )
    elapsed = (time.perf_counter() - t0) * 1000
    statuses = {s.slot_id: s.model_dump() for s in evaluation.slots}
    return {
        "slot_statuses": statuses,
        "evidence_round": state.get("evidence_round", 0) + 1,
        "scope_coverage": evaluation.scope_coverage or "",
        "trace": [new_trace_step(
            "evaluator", status=evaluation.overall_status,
            missing=evaluation.missing, contradiction=evaluation.contradiction,
            ambiguity=evaluation.ambiguity,
            scope_coverage=evaluation.scope_coverage,
            marginal_gain=evaluation.marginal_gain,
            latency_ms=round(elapsed, 1),
        )],
    }


def answer_node(state: AgentState) -> dict:
    import time

    t0 = time.perf_counter()
    metrics = get_metrics()
    # collection questions need a wider final window (parity with Pipeline 2's
    # COLLECTION_TOP_N): counting across sibling events is window-bound
    top_n = config.TOP_N_CHUNKS
    if state.get("question_type") == "collection_aggregation":
        top_n = max(top_n, config.COLLECTION_TOP_N_CHUNKS)
    chunks, graph_paths = build_context(state, top_n=top_n)
    answer, usage = generate_answer(state["question"], chunks, graph_paths, metrics=metrics)
    elapsed = (time.perf_counter() - t0) * 1000

    # Models sometimes cite block numbers ("1", "3") instead of chunk_ids —
    # map numeric citations back to the block's chunk_id.
    id_by_block = {str(i): c["id"] for i, c in enumerate(chunks, 1)}
    citations = [id_by_block.get(c, c) for c in answer.cited_chunk_ids]
    answer = answer.model_copy(update={"cited_chunk_ids": citations})

    return {
        "context_tokens": usage.get("context_tokens", 0),
        "final_answer": answer.model_dump(),
        "final_chunks": chunks,
        "trace": [new_trace_step(
            "answer", citations=citations,
            evidence_sufficient=answer.evidence_sufficient,
            latency_ms=round(elapsed, 1),
        )],
        "stop_reason": state.get("stop_reason") or "answered",
    }


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #

def _route_from_supervisor(state: AgentState) -> str:
    action = state.get("next_action", STOP)
    return {
        GRAPH_SEARCH: "graph_search",
        GRAPH_SCOPE: "scope_search",
        VECTOR_SEARCH: "vector_search",
        COMMUNITY_SEARCH: "community_search",
        REFORMULATE: "evaluator",   # reformulated subquery -> re-evaluate -> re-decide
        STOP: "answer",
    }.get(action, "answer")


def build_agent_graph():
    """Compile the agentic investigation graph (v2 §10.2)."""
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("context_resolver", context_resolver_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("graph_search", _tool_node(GRAPH_SEARCH))
    g.add_node("scope_search", _tool_node(GRAPH_SCOPE))
    g.add_node("vector_search", _tool_node(VECTOR_SEARCH))
    g.add_node("community_search", _tool_node(COMMUNITY_SEARCH))
    g.add_node("evaluator", evaluator_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "context_resolver")
    g.add_edge("context_resolver", "supervisor")
    g.add_conditional_edges("supervisor", _route_from_supervisor)
    for tool in ("graph_search", "scope_search", "vector_search", "community_search"):
        g.add_edge(tool, "evaluator")
    g.add_edge("evaluator", "supervisor")
    g.add_edge("answer", END)
    return g.compile()
