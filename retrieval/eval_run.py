"""Batch eval for Pipeline 1 (Vector RAG) over Data/eval_public.jsonl.

Same scoring as the other pipelines (graphrag.eval_run.score_question) and
same run persistence (shared.run_store) — comparable dashboard rows.

Usage:
    python -m retrieval.eval_run --limit 10
"""

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from retrieval.vector_pipeline import answer_question

__all__ = ["run_vector_eval"]

RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"


def run_vector_eval(limit: int | None = None, output: Path = RESULTS_PATH,
                    eval_set: str = "public") -> dict:
    from graphrag.eval_run import score_question
    from shared.run_store import save_run

    questions = []
    with open(f"Data/eval_{eval_set}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    if limit:
        questions = questions[:limit]

    records, start = [], time.perf_counter()
    for gold in tqdm(questions, desc="vector eval"):
        try:
            result = answer_question(gold["question"])
            records.append(score_question(
                {
                    "answer": result["answer"],
                    "chunks": result["chunks"],
                    "metrics": result["metrics"],
                },
                gold,
            ))
            records[-1]["context_tokens"] = result["context_tokens"]
            records[-1]["retrieval_latency_ms"] = result["retrieval_latency_ms"]
            records[-1]["llm_latency_ms"] = result["llm_latency_ms"]
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
        "avg_context_tokens": round(
            sum(r.get("context_tokens", 0) for r in ok_records) / len(ok_records), 1
        ) if ok_records else 0.0,
    }
    save_run("vector_rag", eval_set, summary, records)
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
    parser.add_argument("--eval-set", default="public", choices=["public", "hidden"])
    args = parser.parse_args()
    run_vector_eval(limit=args.limit, eval_set=args.eval_set)
