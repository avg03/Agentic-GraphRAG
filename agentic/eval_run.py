"""Batch eval for Agentic GraphRAG over Data/eval_public.jsonl.

Reuses graphrag.eval_run's scoring; adds agentic-specific fields
(steps, tools, strategy changes, stop reasons) to each record.

Usage:
    python -m agentic.eval_run --limit 5
    python -m agentic.eval_run --pilot      # questions touching pilot docs
"""

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from agentic.agent import run_agentic

__all__ = ["run_agentic_eval"]

RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"


def run_agentic_eval(limit: int | None = None, pilot_only: bool = False,
                     output: Path = RESULTS_PATH, eval_set: str = "public") -> dict:
    from graphrag.eval_run import score_question

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

    records, start = [], time.perf_counter()
    for gold in tqdm(questions, desc="agentic eval"):
        try:
            result = run_agentic(gold["question"])
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
                    "agent_steps": result.metadata["steps"],
                    "agent_tools": result.metadata["tools_used"],
                    "strategy_changed": result.metadata["strategy_changed"],
                    "stop_reason": result.metadata["stop_reason"],
                    "reason_codes": result.metadata["reason_codes"],
                    "duplicate_chunks_dropped": result.metadata["duplicate_chunks_dropped"],
                }
            )
            records.append(record)
        except Exception as exc:
            records.append({"qid": gold["qid"], "error": str(exc)[:300], "answer_match": False})
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
    print(json.dumps(summary, indent=1))
    print(f"results -> {output}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--eval-set", default="public", choices=["public", "hidden"])
    args = parser.parse_args()
    run_agentic_eval(limit=args.limit, pilot_only=args.pilot, eval_set=args.eval_set)
