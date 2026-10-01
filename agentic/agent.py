"""Agentic GraphRAG driver (doc §14 control loop lives in the compiled graph).

    from agentic.agent import run_agentic

    result = run_agentic("Which athletes from Hungary won medals ... ?")
    print(result.answer)

    python -m agentic.agent "question here"
"""

import time

from agentic._runtime import set_metrics
from agentic.graph import build_agent_graph
from agentic.state import RAGResult
from shared.metrics import Metrics

__all__ = ["run_agentic"]

_compiled = None


def _get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_agent_graph()
    return _compiled


def run_agentic(question: str) -> RAGResult:
    """Run the full agentic investigation for one question."""
    metrics = Metrics()
    start = time.perf_counter()

    set_metrics(metrics)
    graph = _get_graph()
    final = graph.invoke(
        {"question": question},
        config={"recursion_limit": 50},
    )
    latency_ms = (time.perf_counter() - start) * 1000

    answer = final.get("final_answer", {})
    chunks = final.get("final_chunks", [])
    context_tokens = final.get("context_tokens", 0)
    summary = metrics.summary()
    llm = summary.get("llm", {})

    tools_used = [t.get("tool") for t in final.get("trace", []) if t.get("node") == "retrieval"]
    actions = [t.get("action") for t in final.get("trace", []) if t.get("node") == "supervisor"]
    strategy_changed = len(set(tools_used)) > 1
    nodes_invoked = sorted({t.get("node") for t in final.get("trace", []) if t.get("node")})
    timings = summary.get("timings", {})
    time_per_tool = {
        name: agg.get("avg_ms", 0) for name, agg in timings.items()
        if name.startswith("tool_")
    }
    retrieval_latency_ms = sum(agg.get("total_ms", 0) for name, agg in timings.items() if name.startswith("tool_"))
    llm_latency_ms = llm.get("latency_total_ms", 0)
    tokens_per_op = round(llm.get("total_tokens", 0) / max(llm.get("calls", 1), 1), 1)

    return RAGResult(
        pipeline="agentic_graphrag",
        question=question,
        answer=answer.get("answer", ""),
        citations=answer.get("cited_chunk_ids", []),
        retrieved_chunks=chunks,
        context_tokens=context_tokens,
        input_tokens=llm.get("input_tokens", 0),
        output_tokens=llm.get("output_tokens", 0),
        total_tokens=llm.get("total_tokens", 0),
        latency_ms=round(latency_ms, 1),
        retrieval_latency_ms=round(retrieval_latency_ms, 1),
        llm_latency_ms=round(llm_latency_ms, 1),
        evidence_sufficient=answer.get("evidence_sufficient"),
        confidence=answer.get("confidence"),
        metadata={
            "steps": final.get("step_count", 0),
            "tools_used": tools_used,
            "actions": actions,
            "nodes_invoked": nodes_invoked,
            "strategy_changed": strategy_changed,
            "stop_reason": final.get("stop_reason", "unknown"),
            "reason_codes": [t.get("reason_code") for t in final.get("trace", []) if t.get("node") == "supervisor"],
            "time_per_tool": time_per_tool,
            "tokens_per_op": tokens_per_op,
            "duplicate_chunks_dropped": final.get("duplicate_chunks_dropped", 0),
            "trace": final.get("trace", []),
            "metrics": summary,
        },
    )


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+")
    parser.add_argument("--trace", action="store_true", help="Print the full agent trace")
    args = parser.parse_args()

    for question in args.question:
        print(f"\n=== {question}")
        result = run_agentic(question)
        print(f"answer      : {result.answer}")
        md = result.metadata
        print(f"sufficient  : {md['evidence_sufficient']} (confidence {md['confidence']})")
        print(f"citations   : {result.citations}")
        print(
            f"steps={md['steps']} tools={md['tools_used']} "
            f"strategy_changed={md['strategy_changed']} stop={md['stop_reason']}"
        )
        print(
            f"tokens: {result.input_tokens} in / {result.output_tokens} out / "
            f"{result.total_tokens} total | latency {result.latency_ms} ms"
        )
        if args.trace:
            print("trace:")
            for step in md["trace"]:
                print(" ", json.dumps(step, ensure_ascii=False)[:220])


if __name__ == "__main__":
    main()
