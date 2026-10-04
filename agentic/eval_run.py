"""Batch eval for Agentic GraphRAG over Data/eval_public.jsonl.

Reuses graphrag.eval_run's scoring; adds agentic-specific fields
(steps, tools, strategy changes, stop reasons, v2 scope metrics) to each
record. Crash-safe: scored records are checkpointed per question and a
restart resumes from where it stopped.

Usage:
    python -m agentic.eval_run --limit 5
    python -m agentic.eval_run --eval-set hidden
    python -m agentic.eval_run --fresh     # ignore checkpoint, start over
"""

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from agentic.agent import run_agentic

__all__ = ["run_agentic_eval"]

RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"


def _agentic_record(result, gold: dict) -> dict:
    """score_question + the agentic/v2 metric fields (shared by eval + acceptance)."""
    from graphrag.eval_run import score_question

    record = score_question(
        {
            "answer": type(
                "A", (), {
                    "answer": result.answer,
                    "cited_chunk_ids": result.citations,
                    "evidence_sufficient": result.metadata.get("evidence_sufficient"),
                    "confidence": result.metadata.get("confidence"),
                },
            )(),
            "chunks": result.retrieved_chunks,
            "metrics": result.metadata.get("metrics", {}),
        },
        gold,
    )
    record.update(
        {
            "agent_steps": result.metadata.get("steps"),
            "agent_tools": result.metadata.get("tools_used"),
            "strategy_changed": result.metadata.get("strategy_changed"),
            "stop_reason": result.metadata.get("stop_reason"),
            "reason_codes": result.metadata.get("reason_codes", []),
            "duplicate_chunks_dropped": result.metadata.get("duplicate_chunks_dropped", 0),
            # full per-question agent trace (plan -> supervisor -> tools ->
            # evaluator -> answer) for direct evaluation of runs
            "trace": result.metadata.get("trace", []),
            # v2 metrics (architecture v2 §18)
            "question_type": result.metadata.get("question_type", ""),
            "contexts_investigated": result.metadata.get("contexts_investigated", []),
            "context_roots_available": result.metadata.get("context_roots_available", 0),
            "scope_coverage": result.metadata.get("scope_coverage", ""),
            "vector_fallback_count": result.metadata.get("vector_fallback_count", 0),
            "reformulation_count": result.metadata.get("reformulation_count", 0),
            "new_evidence_per_step": result.metadata.get("new_evidence_per_step", 0),
            "evidence_stats": result.metadata.get("evidence_stats", {}),
        }
    )
    return record


def run_agentic_eval(limit: int | None = None, pilot_only: bool = False,
                     output: Path = RESULTS_PATH, eval_set: str = "public",
                     fresh: bool = False) -> dict:
    # Hang watchdog (see graphrag.eval_run.run_eval): stalls raise instead of
    # blocking forever; the per-question retry + checkpoint keep the run alive.
    import socket

    socket.setdefaulttimeout(180)

    from graphrag.eval_run import _append_checkpoint, _checkpoint_file, _load_checkpoint

    questions = []
    with open(f"Data/eval_{eval_set}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if pilot_only:
        from graphrag.extract import load_cached_extractions

        pilot_docs = {r["doc_id"] for r in load_cached_extractions()}
        questions = [q for q in questions if set(q.get("gold_doc_ids", [])) & pilot_docs]
        print(f"{len(questions)} questions touch the pilot docs")
    if limit:
        questions = questions[:limit]

    # crash-safe resume (same checkpoint contract as graphrag.eval_run)
    checkpoint = _checkpoint_file("agentic_graphrag", eval_set)
    done = {} if fresh else _load_checkpoint(checkpoint)
    pending = [q for q in questions if q["qid"] not in done]
    if done:
        print(f"resume: {len(done)} checkpointed, {len(pending)} remaining")
    records_by_qid = dict(done)

    start = time.perf_counter()
    for gold in tqdm(pending, desc="agentic eval"):
        try:
            try:
                result = run_agentic(gold["question"])
                record = _agentic_record(result, gold)
            except Exception as exc:
                # transient hiccup — one retry. TG Cloud hibernation raises a
                # token/500 error and auto-wakes after ~1-2 min, so wait it out
                wait = 90 if ("500" in str(exc) or "token" in str(exc).lower()) else 20
                time.sleep(wait)
                result = run_agentic(gold["question"])
                record = _agentic_record(result, gold)
        except Exception as exc:
            record = {"qid": gold["qid"], "error": str(exc)[:300], "answer_match": False}
        records_by_qid[gold["qid"]] = record
        _append_checkpoint(checkpoint, record)

    records = [records_by_qid[q["qid"]] for q in questions if q["qid"] in records_by_qid]
    wall = time.perf_counter() - start

    ok_records = [r for r in records if "error" not in r]
    matched = sum(1 for r in ok_records if r.get("answer_match"))
    summary = {
        "n_questions": len(records),
        "answer_accuracy": round(matched / len(ok_records), 4) if ok_records else 0.0,
        "wall_time_s": round(wall, 1),
        "avg_total_tokens": round(
            sum(r.get("metrics", {}).get("llm", {}).get("total_tokens", 0) for r in ok_records)
            / len(ok_records), 1) if ok_records else 0.0,
        "avg_agent_steps": round(
            sum(r.get("agent_steps", 0) for r in ok_records) / len(ok_records), 2
        ) if ok_records else 0.0,
        "strategy_change_rate": round(
            sum(1 for r in ok_records if r.get("strategy_changed")) / len(ok_records), 4
        ) if ok_records else 0.0,
        "avg_contexts_investigated": round(
            sum(len(r.get("contexts_investigated", []) or []) for r in ok_records)
            / len(ok_records), 2) if ok_records else 0.0,
        "scope_question_count": sum(
            1 for r in ok_records if r.get("question_type") == "collection_aggregation"
        ),
    }
    from shared.run_store import save_run

    save_run("agentic_graphrag", ("pilot_only_" if pilot_only else "") + eval_set, summary, records, extra_config={
        "n_questions_requested": limit,
    })
    # backward-compatible latest snapshot for quick access
    output.write_text(
        json.dumps({"summary": summary, "records": records}, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    checkpoint.unlink(missing_ok=True)  # run complete; the saved run is authoritative
    print(json.dumps(summary, indent=1))
    print(f"results -> {output}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--eval-set", default="public", choices=["public", "hidden"])
    parser.add_argument("--fresh", action="store_true", help="Ignore any checkpoint and start over")
    args = parser.parse_args()
    run_agentic_eval(limit=args.limit, pilot_only=args.pilot, eval_set=args.eval_set, fresh=args.fresh)
