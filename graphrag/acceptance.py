"""Acceptance gate over Data/eval_v2_classes.jsonl (architecture v2 §21).

Runs the v2 §21 query classes A-G through a pipeline and prints a per-class
pass/fail table. This is the before/after regression gate for the v2
migration: run it before and after each phase; accuracy per class must not
drop.

Usage:
    python -m graphrag.acceptance                 # Pipeline 2 (GraphRAG)
    python -m graphrag.acceptance --agentic       # Pipeline 3 (Agentic)
    python -m graphrag.acceptance --limit 5
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

EVAL_PATH = Path(__file__).resolve().parent.parent / "Data" / "eval_v2_classes.jsonl"

CLASS_ORDER = ["A", "B", "C", "D", "E", "F", "G"]


def load_classes(path: Path = EVAL_PATH) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def run_acceptance(limit: int | None = None, agentic: bool = False,
                   classes: str | None = None) -> dict:
    from graphrag.eval_run import score_question

    questions = load_classes()
    if classes:
        wanted = {c.strip().upper() for c in classes.split(",") if c.strip()}
        questions = [q for q in questions if q.get("v2_class") in wanted]
    if limit:
        questions = questions[:limit]

    records, start = [], time.perf_counter()
    for gold in tqdm(questions, desc="acceptance"):
        try:
            if agentic:
                from agentic.agent import run_agentic

                result = run_agentic(gold["question"])
                adapted = {
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
                }
                record = score_question(adapted, gold)
                # agentic-specific fields (mirror agentic.eval_run)
                record.update(
                    {
                        "agent_steps": result.metadata.get("steps"),
                        "agent_tools": result.metadata.get("tools_used"),
                        "strategy_changed": result.metadata.get("strategy_changed"),
                        "stop_reason": result.metadata.get("stop_reason"),
                        "question_type": result.metadata.get("question_type", ""),
                        "contexts_investigated": result.metadata.get("contexts_investigated", []),
                        "context_roots_available": result.metadata.get("context_roots_available", 0),
                        "scope_coverage": result.metadata.get("scope_coverage", ""),
                        "vector_fallback_count": result.metadata.get("vector_fallback_count", 0),
                        "reformulation_count": result.metadata.get("reformulation_count", 0),
                        "new_evidence_per_step": result.metadata.get("new_evidence_per_step", 0),
                        "evidence_stats": result.metadata.get("evidence_stats", {}),
                        "trace": result.metadata.get("trace", []),
                    }
                )
            else:
                from graphrag.retrieval_pipeline import answer_question

                adapted = answer_question(gold["question"])
                record = score_question(adapted, gold)
        except Exception as exc:
            record = {"qid": gold["qid"], "error": str(exc)[:300], "answer_match": False}
        record["v2_class"] = gold.get("v2_class", "?")
        records.append(record)
    wall = time.perf_counter() - start

    # per-class rollup
    by_class: dict[str, list] = defaultdict(list)
    for r in records:
        by_class[r["v2_class"]].append(r)
    class_summary = {}
    for cls in CLASS_ORDER:
        rows = by_class.get(cls, [])
        if not rows:
            continue
        ok = sum(1 for r in rows if r.get("answer_match"))
        class_summary[cls] = {
            "n": len(rows),
            "correct": ok,
            "accuracy": round(ok / len(rows), 4),
        }
    ok_total = sum(1 for r in records if r.get("answer_match"))
    summary = {
        "n_questions": len(records),
        "answer_accuracy": round(ok_total / len(records), 4) if records else 0.0,
        "by_class": class_summary,
        "wall_time_s": round(wall, 1),
        "pipeline": "agentic_graphrag" if agentic else "graph_rag",
    }

    _print_table(summary, records)

    from shared.run_store import save_run

    save_run(
        "agentic_graphrag" if agentic else "graph_rag",
        "acceptance_v2",
        summary,
        records,
        extra_config={"eval_file": str(EVAL_PATH)},
    )
    return summary


def _print_table(summary: dict, records: list[dict]) -> None:
    print(f"\n== Acceptance ({summary['pipeline']}) — accuracy {summary['answer_accuracy']:.0%} ==")
    for cls, stats in summary["by_class"].items():
        print(f"  Class {cls}: {stats['correct']}/{stats['n']} ({stats['accuracy']:.0%})")
    for r in records:
        mark = "PASS" if r.get("answer_match") else "MISS"
        q = r.get("question", r.get("qid", "?"))[:70]
        pred = str(r.get("predicted", r.get("error", "")))[:60]
        print(f"  [{r['v2_class']}] {mark}  {q} -> {pred}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--agentic", action="store_true", help="Run Pipeline 3 instead of Pipeline 2")
    parser.add_argument("--classes", default=None, help="Comma-separated v2 classes to run (e.g. C,D)")
    args = parser.parse_args()
    run_acceptance(limit=args.limit, agentic=args.agentic, classes=args.classes)


if __name__ == "__main__":
    main()
