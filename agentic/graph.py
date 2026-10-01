"""LangGraph wiring (doc §19): planner → supervisor → tools → evaluator → … → answer."""

from langgraph.graph import StateGraph, END

from agentic import config
from agentic.state import AgentState, new_trace_step
from agentic._runtime import get_metrics
from agentic.planner import plan_question
from agentic.supervisor import decide_next_action
from agentic.evaluator import evaluate_evidence
from agentic.context_builder import build_context
from agentic.answer import generate_answer

__all__ = ["build_agent_graph", "GRAPH_SEARCH", "VECTOR_SEARCH", "COMMUNITY_SEARCH", "REFORMULATE", "STOP"]

GRAPH_SEARCH = "GRAPH_SEARCH"
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
        "trace": [new_trace_step(
            "planner", action="decompose",
            reason="multiple answer slots" if len(plan.answer_slots) > 1 else "single slot",
            slots=[s.id for s in plan.answer_slots], latency_ms=round(elapsed, 1),
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
        "vector_calls": state.get("vector_calls", 0),
        "max_vector": config.MAX_VECTOR_CALLS,
        "community_calls": state.get("community_calls", 0),
        "max_community": config.MAX_COMMUNITY_CALLS,
    }
    decision, _ = decide_next_action(
        state["question"], state.get("slots", []),
        state.get("slot_statuses", {}), state.get("trace", []),
        budget, metrics=metrics,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    update = {
        "next_action": decision.action,
        "target_slot": decision.target_slot,
        "reason_code": decision.reason_code,
        "trace": [new_trace_step(
            "supervisor", action=decision.action,
            reason_code=decision.reason_code, target=decision.target_slot,
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


def _tool_node(tool_name: str):
    def node(state: AgentState) -> dict:
        import time

        t0 = time.perf_counter()
        subquery = state.get("subquery") or state["question"]
        metrics = get_metrics()
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
            outcome = f"{len(new_chunks)} chunks, entities={result.get('entities', [])}"
        elif tool_name == VECTOR_SEARCH:
            from agentic.tools import vector_search

            result = vector_search(subquery)
            new_chunks = result.get("chunks", [])
            update = {"chunks": new_chunks, "vector_calls": state.get("vector_calls", 0) + 1}
            outcome = f"{len(new_chunks)} chunks"
        else:
            from agentic.tools import community_search

            result = community_search(subquery)
            update = {"community_calls": state.get("community_calls", 0) + 1}
            outcome = f"{len(result.get('communities', []))} doc communities"
        elapsed = (time.perf_counter() - t0) * 1000

        trace = [new_trace_step(
            "retrieval", step=state.get("step_count", 0) + 1,
            tool=tool_name, target_slot=state.get("target_slot", ""),
            subquery=subquery, result=outcome, latency_ms=round(elapsed, 1),
        )]
        update["trace"] = trace
        update["step_count"] = state.get("step_count", 0) + 1
        update["seen_queries"] = state.get("seen_queries", []) + [(tool_name, subquery)]
        update["last_tool_result"] = outcome
        if metrics is not None:
            metrics.record_elapsed(f"tool_{tool_name.lower()}", elapsed)
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
    )
    elapsed = (time.perf_counter() - t0) * 1000
    statuses = {s.slot_id: s.model_dump() for s in evaluation.slots}
    return {
        "slot_statuses": statuses,
        "evidence_round": state.get("evidence_round", 0) + 1,
        "trace": [new_trace_step(
            "evaluator", status=evaluation.overall_status,
            missing=evaluation.missing, contradiction=evaluation.contradiction,
            ambiguity=evaluation.ambiguity, latency_ms=round(elapsed, 1),
        )],
    }


def answer_node(state: AgentState) -> dict:
    import time

    t0 = time.perf_counter()
    metrics = get_metrics()
    chunks, graph_paths = build_context(state, top_n=config.TOP_N_CHUNKS)
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
        VECTOR_SEARCH: "vector_search",
        COMMUNITY_SEARCH: "community_search",
        REFORMULATE: "evaluator",   # reformulated subquery -> re-evaluate -> re-decide
        STOP: "answer",
    }.get(action, "answer")


def build_agent_graph():
    """Compile the agentic investigation graph."""
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("graph_search", _tool_node(GRAPH_SEARCH))
    g.add_node("vector_search", _tool_node(VECTOR_SEARCH))
    g.add_node("community_search", _tool_node(COMMUNITY_SEARCH))
    g.add_node("evaluator", evaluator_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "supervisor")
    g.add_conditional_edges("supervisor", _route_from_supervisor)
    for tool in ("graph_search", "vector_search", "community_search"):
        g.add_edge(tool, "evaluator")
    g.add_edge("evaluator", "supervisor")
    g.add_edge("answer", END)
    return g.compile()
