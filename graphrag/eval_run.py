"""Batch eval over Data/eval_public.jsonl for Pipeline 2.

Per question: run answer_question, score correctness, record metrics.
Correctness:
    answer_match — normalized predicted answer matches any gold answer string
    gold_doc_coverage — fraction of cited chunks whose doc_id is in gold_doc_ids
Aggregate: accuracy, avg latency/tokens -> printed + saved to JSON.

Pilot note: the graph currently covers only the pilot docs; use --pilot to
restrict to questions whose gold_doc_ids intersect the ingested docs.

Usage:
    python -m graphrag.eval_run --limit 10
    python -m graphrag.eval_run --pilot          # only graph-covered questions
"""

import argparse
import json
import re
import time
from pathlib import Path

from tqdm import tqdm

from graphrag.retrieval_pipeline import answer_question

__all__ = ["normalize_answer", "score_question", "run_eval"]

RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"


def normalize_answer(text: str) -> str:
    """Loose normalization for answer matching."""
    lowered = (text or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9 ]", " ", lowered)
    return " ".join(lowered.split())


def score_question(result: dict, gold: dict) -> dict:
    answer = result["answer"]
    predicted = normalize_answer(answer.answer)
    gold_answers = [normalize_answer(a) for a in gold.get("answer", [])]
    # Word-boundary match (a bare substring check makes short gold answers
    # like "20" falsely match "2004"). Substring only for longer golds.
    def _matches(gold: str) -> bool:
        if not gold:
            return False
        if predicted == gold:
            return True
        if len(gold) > 3 and gold in predicted:
            return True
        return re.search(rf"\b{re.escape(gold)}\b", predicted) is not None

    answer_match = any(_matches(g) for g in gold_answers)

    gold_docs = set(gold.get("gold_doc_ids", []))
    cited_docs = {
        c.get("metadata", {}).get("doc_id", "")
        for c in result["chunks"]
    }
    coverage = len(cited_docs & gold_docs) / len(gold_docs) if gold_docs else None

    metrics_obj = result.get("metrics")
    metrics_summary = metrics_obj.summary() if hasattr(metrics_obj, "summary") else metrics_obj

    return {
        "qid": gold["qid"],
        "question": gold["question"],
        "qtype": gold.get("qtype"),
        "predicted": answer.answer,
        "gold": gold.get("answer", []),
        "answer_match": answer_match,
        "gold_doc_coverage": coverage,
        "evidence_sufficient": answer.evidence_sufficient,
        "confidence": answer.confidence,
        "n_seeds": len(result.get("seeds", []) or []),
        "n_graph_chunks": result.get("rank_info", {}).get("graph_chunks", 0) if isinstance(result.get("rank_info"), dict) else 0,
        "n_dense_topup": result.get("rank_info", {}).get("dense_topup", 0) if isinstance(result.get("rank_info"), dict) else 0,
        "n_final_chunks": len(result["chunks"]),
        "cited_chunk_ids": answer.cited_chunk_ids,
        "metrics": metrics_summary,
    }


def run_eval(
    limit: int | None = None,
    pilot_only: bool = False,
    output: Path = RESULTS_PATH,
    eval_set: str = "public",
) -> dict:
    from ingestion.preprocessing import load_corpus

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

    records = []
    start = time.perf_counter()
    for gold in tqdm(questions, desc="eval"):
        try:
            result = answer_question(gold["question"])
            records.append(score_question(result, gold))
        except Exception as exc:
            records.append(
                {"qid": gold["qid"], "error": str(exc)[:300], "answer_match": False}
            )
    wall = time.perf_counter() - start

    matched = sum(1 for r in records if r.get("answer_match"))
    summary = {
        "n_questions": len(records),
        "answer_accuracy": round(matched / len(records), 4) if records else 0.0,
        "wall_time_s": round(wall, 1),
        "avg_total_tokens": round(
            sum(r.get("metrics", {}).get("llm", {}).get("total_tokens", 0) for r in records)
            / len(records), 1
        ) if records else 0.0,
        "avg_latency_ms": round(
            sum(
                sum(t.get("total_ms", 0) for t in r.get("metrics", {}).get("timings", {}).values())
                for r in records
            ) / len(records), 1
        ) if records else 0.0,
    }

    from shared.run_store import save_run

    save_run("graph_rag", ("pilot_only_" if pilot_only else "") + eval_set, summary, records, extra_config={
        "n_questions_requested": limit,
        "results_dir_override": str(output) if str(output) != "graphrag/eval_results.json" else None,
    })
    # backward-compatible latest snapshot for quick access
    output.write_text(
        json.dumps({"summary": summary, "records": records}, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=1))
    print(f"results -> {output}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pilot", action="store_true", help="Only questions touching pilot docs")
    parser.add_argument("--eval-set", default="public", choices=["public", "hidden"])
    args = parser.parse_args()
    run_eval(limit=args.limit, pilot_only=args.pilot, eval_set=args.eval_set)


if __name__ == "__main__":
    main()
