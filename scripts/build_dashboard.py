#!/usr/bin/env python3
"""Build a self-contained dashboard (dashboard/index.html) from the newest saved
eval run per pipeline per set (public + hidden) under results/.

- Reads: results/<pipeline>/{public,hidden}_*.json for pipelines
         vector_rag, graph_rag, agentic_graphrag, plus Data/eval_public.jsonl
         (qid -> qtype join).
- Writes: dashboard/index.html — single file, all CSS inline, data embedded as a
         JSON blob, no external resources, opens directly via file://.

Stdlib only. If a pipeline has no run for a set yet (e.g. the agentic v2 hidden
run still in flight), the dashboard uses what exists and notes it.
"""

import html
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "Data"
OUT_PATH = REPO_ROOT / "dashboard" / "index.html"

PIPELINES = {
    "vector_rag": {"label": "Vector RAG (final)", "color": "#4e9aef"},
    "graph_rag": {"label": "GraphRAG (final)", "color": "#3fb98c"},
    "agentic_graphrag": {"label": "Agentic GraphRAG v2", "color": "#e0a13a"},
}
MODEL_NAME = "gpt-5-mini (Azure OpenAI)"
CORPUS_DESC = "2,951 documents / ~17k chunks"

TS_RE_LEN = len("_YYYYMMDD_HHMMSS.json")


def newest_run(pipeline: str, prefix: str):
    """Newest results/<pipeline>/<prefix>_*.json -> (path, doc) or None."""
    pdir = RESULTS_DIR / pipeline
    if not pdir.is_dir():
        return None
    candidates = []
    for p in pdir.glob(f"{prefix}_*.json"):
        key = p.name[-TS_RE_LEN + 1:-5] if p.name.endswith(".json") else p.name
        candidates.append((key, p))
    if not candidates:
        return None
    path = max(candidates, key=lambda c: c[0])[1]
    try:
        with path.open(encoding="utf-8") as f:
            return path, json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def record_latencies(pipeline: str, records: list):
    """Per-record (total_ms, retrieval_ms, llm_ms) using each pipeline's schema."""
    out = []
    for r in records:
        m = r.get("metrics") or {}
        timings = (m.get("timings") or {})
        timings_sum = sum(t.get("total_ms", 0.0) for t in timings.values()) if timings else 0.0
        llm_ms = float((m.get("llm") or {}).get("latency_total_ms", 0.0) or 0.0)
        if pipeline == "vector_rag":
            ret = float(r.get("retrieval_latency_ms") or timings.get("retrieval", {}).get("total_ms", 0.0))
            llm = float(r.get("llm_latency_ms") or llm_ms)
            total = ret + llm
        elif pipeline == "graph_rag":
            # stage timings include the LLM stages; split via metrics.llm
            total = timings_sum or float(r.get("avg_latency_ms") or 0.0)
            llm = llm_ms
            ret = max(total - llm, 0.0)
        else:  # agentic_graphrag: tool timings exclude agent LLM calls
            ret = timings_sum
            llm = llm_ms
            total = ret + llm
        out.append((total, ret, llm))
    return out


def is_v2_run(pipeline: str, doc: dict) -> bool:
    """v2 runs record scope/retrieval-mode info per question."""
    if pipeline == "vector_rag":
        return True
    records = doc.get("records") or []
    return bool(records) and any(
        ("retrieval_mode" in r or "question_type" in r) for r in records
    )


def summarize_run(pipeline: str, doc: dict):
    records = doc.get("records") or []
    summary = doc.get("summary") or {}
    lat = record_latencies(pipeline, records)
    totals = [t for t, _, _ in lat] or [0.0]
    rets = [r for _, r, _ in lat]
    llms = [l for _, _, l in lat]
    n = max(len(records), 1)
    row = {
        "pipeline": pipeline,
        "label": PIPELINES[pipeline]["label"],
        "run_id": doc.get("run_id"),
        "source_file": doc.get("run_id"),
        "n_questions": len(records),
        "v2_run": is_v2_run(pipeline, doc),
        "accuracy": summary.get("answer_accuracy"),
        "avg_tokens": summary.get("avg_total_tokens"),
        "avg_latency_s": statistics.mean(totals) / 1000.0 if totals else None,
        "p50_latency_s": statistics.median(totals) / 1000.0 if totals else None,
        "avg_retrieval_s": (statistics.mean(rets) / 1000.0) if rets else None,
        "avg_llm_s": (statistics.mean(llms) / 1000.0) if llms else None,
        "avg_steps": None,
        "strategy_change_rate": None,
    }
    if pipeline == "agentic_graphrag":
        steps = [r.get("agent_steps") for r in records if isinstance(r.get("agent_steps"), (int, float))]
        row["avg_steps"] = round(statistics.mean(steps), 2) if steps else summary.get("avg_agent_steps")
        sc = [1 if r.get("strategy_changed") else 0 for r in records]
        row["strategy_change_rate"] = round(sum(sc) / len(sc), 2) if sc else summary.get("strategy_change_rate")
    return row


def qtype_breakdown(per_pipeline_public):
    """qtype accuracy per pipeline on the public set.

    qtype is joined from Data/eval_public.jsonl by qid (vector/graph pipelines);
    falls back to record qtype / question_type (agentic v2) when needed.
    """
    qtype_by_qid = {}
    eval_file = DATA_DIR / "eval_public.jsonl"
    if eval_file.exists():
        with eval_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qtype_by_qid[rec["qid"]] = rec.get("qtype")
    breakdown = {}
    for pipeline, doc in per_pipeline_public.items():
        if doc is None:
            continue
        acc = {}
        for r in doc.get("records") or []:
            qtype = qtype_by_qid.get(r.get("qid")) or r.get("qtype") or r.get("question_type") or "unknown"
            acc.setdefault(qtype, []).append(1 if r.get("answer_match") else 0)
        breakdown[pipeline] = {
            q: round(sum(v) / len(v), 3) for q, v in sorted(acc.items())
        }
    qtypes = sorted({q for d in breakdown.values() for q in d})
    return qtypes, breakdown


def fmt(v, nd=1, suffix=""):
    if v is None:
        return "&mdash;"
    return f"{v:.{nd}f}{suffix}"


def pct(v):
    return "&mdash;" if v is None else f"{v * 100:.0f}%"


def comparison_table(rows, show_accuracy, notes=None):
    head = (
        "<tr><th>Pipeline</th><th>Questions</th>"
        + ("<th>Accuracy</th>" if show_accuracy else "")
        + "<th>Avg tokens / q</th><th>Avg latency</th><th>p50 latency</th>"
        + "<th>Retrieval / LLM split</th><th>Steps / q</th><th>Strategy-change rate</th></tr>"
    )
    body = []
    for row in rows:
        split = f"{fmt(row['avg_retrieval_s'])}s / {fmt(row['avg_llm_s'])}s"
        steps = fmt(row["avg_steps"], 2) if pipeline_is_agentic(row) else "&mdash;"
        scr = pct(row["strategy_change_rate"]) if pipeline_is_agentic(row) else "&mdash;"
        acc = pct(row["accuracy"]) if show_accuracy else "&mdash;"
        note = ""
        if notes and row["pipeline"] in notes:
            note = f'<div class="note">{html.escape(notes[row["pipeline"]])}</div>'
        body.append(
            "<tr>"
            f'<td class="pl">{html.escape(row["label"])}{note}</td>'
            f"<td>{row['n_questions']}</td>"
            + (f"<td>{acc}</td>" if show_accuracy else "")
            + f"<td>{fmt(row['avg_tokens'], 0)}</td>"
            f"<td>{fmt(row['avg_latency_s'])}s</td>"
            f"<td>{fmt(row['p50_latency_s'])}s</td>"
            f"<td>{split}</td>"
            f"<td>{steps}</td>"
            f"<td>{scr}</td>"
            "</tr>"
        )
    return f'<table class="cmp"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'


def pipeline_is_agentic(row):
    return row["pipeline"] == "agentic_graphrag"


def build_html(public_rows, hidden_rows, qtypes, breakdown, hidden_notes):
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data_blob = {
        "generated_at": generated,
        "model": MODEL_NAME,
        "corpus": CORPUS_DESC,
        "public": public_rows,
        "hidden": hidden_rows,
        "qtype_accuracy_public": breakdown,
    }

    # --- qtype chart (CSS bars, one group per qtype) ---
    chart_rows = []
    for q in qtypes:
        bars = []
        for pipeline, meta in PIPELINES.items():
            v = breakdown.get(pipeline, {}).get(q)
            w = 0 if v is None else max(int(round(v * 100)), 2)
            bars.append(
                f'<div class="bar-row"><span class="bar-label">{html.escape(meta["label"])}</span>'
                f'<div class="bar-track"><div class="bar" style="width:{w}%;background:{meta["color"]}"></div></div>'
                f'<span class="bar-val">{pct(v)}</span></div>'
            )
        chart_rows.append(f'<div class="qtype-group"><div class="qtype-name">{html.escape(q)}</div>{"".join(bars)}</div>')

    qtype_table_cells = []
    for q in qtypes:
        cells = [f"<td>{html.escape(q)}</td>"]
        for pipeline in PIPELINES:
            v = breakdown.get(pipeline, {}).get(q)
            n_q = ""
            cells.append(f"<td>{pct(v)}{n_q}</td>")
        qtype_table_cells.append("".join(cells))

    hidden_note_html = ""
    if hidden_notes:
        items = "".join(f"<li>{html.escape(n)}</li>" for n in hidden_notes)
        hidden_note_html = f'<ul class="notes">{items}</ul>'

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GraphRAG Hackathon &mdash; Pipeline Comparison Dashboard</title>
<style>
  :root {{ --bg:#14171c; --card:#1d222a; --line:#2c333e; --fg:#e6e9ee; --muted:#9aa4b2; --accent:#4e9aef; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:32px 20px 60px; background:var(--bg); color:var(--fg);
         font:15px/1.55 "Segoe UI", system-ui, -apple-system, sans-serif; }}
  .wrap {{ max-width:1080px; margin:0 auto; }}
  h1 {{ font-size:26px; margin:0 0 4px; }}
  h2 {{ font-size:19px; margin:36px 0 10px; border-bottom:1px solid var(--line); padding-bottom:6px; }}
  .meta {{ color:var(--muted); font-size:14px; }}
  .meta b {{ color:var(--fg); font-weight:600; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin-top:14px; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; }}
  th, td {{ padding:9px 12px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }}
  th {{ color:var(--muted); font-weight:600; font-size:12.5px; text-transform:uppercase; letter-spacing:.04em; }}
  td.pl {{ font-weight:600; white-space:normal; min-width:190px; }}
  tr:last-child td {{ border-bottom:none; }}
  .note {{ font-weight:400; font-size:12.5px; color:#e0a13a; margin-top:3px; }}
  ul.notes {{ color:var(--muted); font-size:13.5px; margin:10px 0 0; }}
  .qtype-group {{ margin:14px 0; }}
  .qtype-name {{ font-weight:600; margin-bottom:5px; }}
  .bar-row {{ display:flex; align-items:center; margin:3px 0; }}
  .bar-label {{ width:170px; flex:none; color:var(--muted); font-size:13px; }}
  .bar-track {{ flex:1; background:#242b35; border-radius:4px; height:14px; overflow:hidden; }}
  .bar {{ height:100%; border-radius:4px; }}
  .bar-val {{ width:52px; flex:none; text-align:right; font-size:13px; }}
  table.qt td, table.qt th {{ white-space:normal; }}
  .v2-list {{ margin:8px 0 0; padding-left:22px; }}
  .v2-list li {{ margin:6px 0; }}
  code {{ background:#242b35; padding:1px 6px; border-radius:4px; font-size:13px; }}
  .footer {{ margin-top:44px; color:var(--muted); font-size:13px; border-top:1px solid var(--line); padding-top:14px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>GraphRAG Hackathon &mdash; Pipeline Comparison Dashboard</h1>
  <div class="meta">
    Model: <b>{html.escape(MODEL_NAME)}</b> &nbsp;&middot;&nbsp;
    Corpus: <b>{CORPUS_DESC}</b> &nbsp;&middot;&nbsp;
    Generated: <b>{generated}</b> &nbsp;&middot;&nbsp;
    Branch: <b>version_2.0</b>
  </div>

  <h2>Public eval set &mdash; 100 questions (ground truth available)</h2>
  <div class="card">
    {comparison_table(public_rows, show_accuracy=True)}
  </div>

  <h2>Hidden eval set &mdash; 50 questions (runtime metrics; scoring done separately)</h2>
  <div class="card">
    {comparison_table(hidden_rows, show_accuracy=False, notes=hidden_notes)}
    {hidden_note_html}
  </div>

  <h2>Per-question-type accuracy &mdash; public set</h2>
  <div class="card">
    {"".join(chart_rows)}
    <table class="qt" style="margin-top:18px">
      <thead><tr><th>Question type</th><th>Vector RAG (final)</th><th>GraphRAG (final)</th><th>Agentic GraphRAG v2</th></tr></thead>
      <tbody>{"".join(f"<tr>{c}</tr>" for c in qtype_table_cells)}</tbody>
    </table>
    <div class="meta" style="margin-top:10px">Question types joined from <code>Data/eval_public.jsonl</code> by <code>qid</code>
    (agentic v2 also records its own finer-grained <code>question_type</code> used by the scope router).</div>
  </div>

  <h2>What changed in v2</h2>
  <div class="card">
    <ul class="v2-list">
      <li><b>Hierarchy-aware context index</b> (<code>graphrag/context_index.py</code>): entity contexts are indexed along the
      graph hierarchy so lookups resolve which subgraph a phrase belongs to, not just that it exists.</li>
      <li><b>Scope traversal</b> (<code>graphrag/scope.py</code> + <code>traverse.py</code>): for set/aggregation questions the
      traversal identifies the full scope of matching entities (e.g. all events of a discipline) instead of a handful of seeds,
      and pulls the union of their chunks.</li>
      <li><b>New agent action <code>GRAPH_SCOPE</code></b> (Pipeline 3): the agent can request a scope expansion when it detects
      an under-covered collection question; a coverage gate decides whether collected evidence is sufficient before answering.</li>
      <li><b>Pipeline 2 (GraphRAG) ships with the v2 scope-capable code path</b> &mdash; the same retrieval engine powers both
      the fixed pipeline and the agent's <code>GRAPH_SCOPE</code> tool.</li>
    </ul>
  </div>

  <div class="footer">
    Regenerate with <code>python scripts/build_dashboard.py</code> after new runs land under <code>results/</code>.
    Hidden-set predictions: <code>results/hidden_predictions/</code> (built by <code>scripts/build_hidden_predictions.py</code>).
  </div>
</div>
<script type="application/json" id="dashboard-data">{json.dumps(data_blob, ensure_ascii=False)}</script>
</body>
</html>
"""
    return doc


def main():
    public_docs, hidden_docs, public_rows, hidden_rows = {}, {}, [], []
    hidden_notes = []

    for pipeline in PIPELINES:
        pub = newest_run(pipeline, "public")
        hid = newest_run(pipeline, "hidden")
        public_docs[pipeline] = pub[1] if pub else None
        hidden_docs[pipeline] = hid[1] if hid else None

        if pub:
            row = summarize_run(pipeline, pub[1])
            row["source_file"] = f"results/{pipeline}/{pub[0].name}"
            public_rows.append(row)
        else:
            print(f"[warn] {pipeline}: no public run found", file=sys.stderr)

        if hid:
            row = summarize_run(pipeline, hid[1])
            row["source_file"] = f"results/{pipeline}/{hid[0].name}"
            hidden_rows.append(row)
            if pipeline == "agentic_graphrag" and not row["v2_run"]:
                hidden_notes.append(
                    "Agentic GraphRAG hidden run on disk is the v1 agent (run "
                    f"{row['run_id']}); the v2 hidden run is still in flight. "
                    "Re-run scripts/build_dashboard.py after it finishes."
                )
        else:
            hidden_notes.append(
                f"{PIPELINES[pipeline]['label']}: no hidden run available yet."
            )

    qtypes, breakdown = qtype_breakdown(public_docs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    page = build_html(public_rows, hidden_rows, qtypes, breakdown, hidden_notes)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        f.write(page)

    print(f"[ok] wrote {OUT_PATH}")
    for row in public_rows:
        acc = "n/a" if row["accuracy"] is None else f"{row['accuracy'] * 100:.0f}%"
        print(f"     public  {row['pipeline']:<18} acc={acc:<5} tokens={row['avg_tokens']} run={row['run_id']}")
    for row in hidden_rows:
        print(f"     hidden  {row['pipeline']:<18} tokens={row['avg_tokens']} run={row['run_id']}")
    for note in hidden_notes:
        print(f"     note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
