#!/usr/bin/env python3
"""Build consolidated hidden-set prediction files from the newest hidden eval run
of each pipeline.

Reads:   results/<pipeline>/hidden_*.json   (pipelines: vector_rag, graph_rag, agentic_graphrag)
Writes:  results/hidden_predictions/<pipeline>_hidden_predictions.json
         results/hidden_predictions/all_pipelines_hidden_predictions.json
         results/hidden_predictions/hidden_predictions.csv

Each prediction entry carries, for each of the 50 hidden questions, the generated
answer, token usage and (for the agentic pipeline) the full agent trace:

    {qid, question, predicted_answer, cited_chunk_ids, evidence_sufficient,
     confidence, total_tokens, input_tokens, output_tokens,
     agent_steps, agent_tools, stop_reason, trace}

- Token fields come from record["metrics"]["llm"].
- trace comes from record.get("trace", []) (agentic eval persists traces; older
  records without a trace get an empty list).
- Pipelines 1 and 2 additionally carry a "stage_timings" field (from
  record["metrics"]["timings"]) as their trace equivalent.

Stdlib only. A pipeline with no hidden run yet (e.g. an eval still in flight)
is skipped gracefully. NOTE: the v2 agentic hidden run is re-run with trace
capture; re-run this script after it completes to fold it in (the script prints
a reminder when the current agentic hidden run has no traces).
"""

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR / "hidden_predictions"
PIPELINES = ["vector_rag", "graph_rag", "agentic_graphrag"]
AGENTIC = "agentic_graphrag"

TS_RE = re.compile(r"_(\d{8}_\d{6})\.json$")

CSV_COLUMNS = ["pipeline", "qid", "question", "predicted_answer", "tokens", "agent_steps", "stop_reason"]


def newest_run(pipeline: str, prefix: str):
    """Return (path, doc) of the newest results/<pipeline>/<prefix>_*.json, or None."""
    pdir = RESULTS_DIR / pipeline
    if not pdir.is_dir():
        return None
    candidates = []
    for p in pdir.glob(f"{prefix}_*.json"):
        m = TS_RE.search(p.name)
        key = m.group(1) if m else "0000_000000"
        candidates.append((key, p))
    if not candidates:
        return None
    path = max(candidates, key=lambda c: c[0])[1]
    try:
        with path.open(encoding="utf-8") as f:
            return path, json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [skip] {pipeline}: could not read {path.name}: {exc}", file=sys.stderr)
        return None


def build_predictions(pipeline: str, records: list):
    predictions = []
    any_trace = False
    for r in records:
        llm = (r.get("metrics") or {}).get("llm") or {}
        trace = r.get("trace") or []
        if trace:
            any_trace = True
        entry = {
            "qid": r.get("qid"),
            "question": r.get("question"),
            "predicted_answer": r.get("predicted"),
            "cited_chunk_ids": r.get("cited_chunk_ids") or [],
            "evidence_sufficient": r.get("evidence_sufficient"),
            "confidence": r.get("confidence"),
            "total_tokens": llm.get("total_tokens"),
            "input_tokens": llm.get("input_tokens"),
            "output_tokens": llm.get("output_tokens"),
            "agent_steps": r.get("agent_steps"),
            "agent_tools": r.get("agent_tools") or [],
            "stop_reason": r.get("stop_reason"),
            "trace": trace,
        }
        if pipeline != AGENTIC:
            timings = (r.get("metrics") or {}).get("timings") or {}
            entry["stage_timings"] = {
                stage: t.get("total_ms") for stage, t in timings.items()
            } if timings else {}
        predictions.append(entry)
    return predictions, any_trace


def build_pipeline_doc(pipeline: str):
    found = newest_run(pipeline, "hidden")
    if found is None:
        return None
    path, doc = found
    records = doc.get("records") or []
    predictions, any_trace = build_predictions(pipeline, records)
    model = (doc.get("config") or {}).get("model")
    doc_out = {
        "pipeline": pipeline,
        "run_id": doc.get("run_id") or path.stem,
        "model": model,
        "n_questions": len(predictions),
        "source_file": f"results/{pipeline}/{path.name}",
        "predictions": predictions,
    }
    if pipeline == AGENTIC and not any_trace:
        doc_out["trace_note"] = (
            "This hidden run was produced before trace capture was added; agent "
            "traces are empty. Re-run scripts/build_hidden_predictions.py after "
            "the v2 agentic hidden run (with traces) completes."
        )
    return doc_out


def write_csv(pipeline_docs: list):
    csv_path = OUT_DIR / "hidden_predictions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for doc in pipeline_docs:
            for p in doc["predictions"]:
                writer.writerow(
                    {
                        "pipeline": doc["pipeline"],
                        "qid": p["qid"],
                        "question": p["question"],
                        "predicted_answer": p["predicted_answer"],
                        "tokens": p["total_tokens"] if p["total_tokens"] is not None else "",
                        "agent_steps": p["agent_steps"] if p["agent_steps"] is not None else "",
                        "stop_reason": p["stop_reason"] or "",
                    }
                )
    return csv_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated, skipped = [], []
    pipeline_docs = []

    for pipeline in PIPELINES:
        doc = build_pipeline_doc(pipeline)
        if doc is None:
            skipped.append(pipeline)
            print(f"[skip] {pipeline}: no hidden run found under results/{pipeline}/ yet")
            continue
        out_path = OUT_DIR / f"{pipeline}_hidden_predictions.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        pipeline_docs.append(doc)
        generated.append(out_path)
        has_traces = "trace_note" not in doc
        print(
            f"[ok]   {pipeline}: {doc['n_questions']} predictions from {doc['source_file']} "
            f"-> {out_path}" + ("" if has_traces else " (no agent traces in this run)")
        )

    if pipeline_docs:
        combined = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "description": (
                "Hidden-set predictions (answer, tokens, agent trace) from the newest "
                "saved hidden run of each pipeline."
            ),
            "pipelines": pipeline_docs,
        }
        combined_path = OUT_DIR / "all_pipelines_hidden_predictions.json"
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        generated.append(combined_path)
        print(f"[ok]   combined -> {combined_path}")

        csv_path = write_csv(pipeline_docs)
        generated.append(csv_path)
        print(f"[ok]   csv      -> {csv_path}")

    print(f"\nGenerated {len(generated)} file(s): {', '.join(str(g.relative_to(REPO_ROOT)) for g in generated) or 'none'}")
    if skipped:
        print(f"Skipped (no hidden run yet): {', '.join(skipped)}")
    agentic_doc = next((d for d in pipeline_docs if d["pipeline"] == AGENTIC), None)
    if agentic_doc is None or "trace_note" in agentic_doc:
        print(
            "HINT: the v2 agentic hidden run is being re-run WITH trace capture. "
            "When it finishes, re-run scripts/build_hidden_predictions.py and "
            "scripts/build_dashboard.py to include it."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
