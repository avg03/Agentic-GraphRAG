"""Run persistence: every eval run is saved as a separate timestamped document.

Layout:
    results/<pipeline>/<eval_set>_<timestamp>.json

Each document is self-describing (pipeline, eval set, model/provider, config,
summary, per-question records) so runs can be compared later — e.g. the same
eval set across models, or model-vs-model dashboards — without any run
overwriting another.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from shared.llm_client import LLM_PROVIDER, DEFAULT_MODEL

__all__ = ["save_run", "load_run", "list_runs"]

_RESULTS_ROOT = Path(os.getenv("RESULTS_DIR", "results"))


def _active_model() -> str:
    """The model that actually served the calls in this run."""
    return DEFAULT_MODEL


def save_run(
    pipeline: str,
    eval_set: str,
    summary: dict,
    records: list[dict],
    extra_config: dict | None = None,
) -> Path:
    """Persist one eval run. Returns the written file path."""
    out_dir = _RESULTS_ROOT / pipeline
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{eval_set}_{stamp}.json"

    doc = {
        "run_id": f"{pipeline}/{eval_set}/{stamp}",
        "pipeline": pipeline,
        "eval_set": eval_set,
        "saved_at": datetime.now().isoformat(),
        "config": {
            "llm_provider": LLM_PROVIDER,
            "model": _active_model(),
            **(extra_config or {}),
        },
        "summary": summary,
        "records": records,
    }
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"run saved -> {path}")
    return path


def load_run(path: str | Path) -> dict:
    """Load a saved run document."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_runs(pipeline: str | None = None) -> list[Path]:
    """List saved run files (newest first), optionally for one pipeline."""
    if pipeline:
        root = _RESULTS_ROOT / pipeline
        files = root.glob("*.json") if root.exists() else []
    else:
        files = _RESULTS_ROOT.glob("*/*.json")
    return sorted(files, reverse=True)
