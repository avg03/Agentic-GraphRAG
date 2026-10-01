"""Run ALL six evaluations sequentially on the configured model, saving each.

Order (public first, then hidden; lightest pipeline first so early results
land quickly):
    1. vector_rag  public (100)      4. vector_rag  hidden (50)
    2. graph_rag   public (100)      5. graph_rag   hidden (50)
    3. agentic     public (100)      6. agentic     hidden (50)

Every run is saved via shared.run_store -> results/<pipeline>/<set>_<ts>.json.
Progress is appended to eval_all_progress.log and printed.
"""

import time
import traceback

from shared.run_store import save_run

SETS = ["public", "hidden"]
PIPELINES = ["vector_rag", "graph_rag", "agentic_graphrag"]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open("eval_all_progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_one(pipeline: str, eval_set: str) -> dict | None:
    if pipeline == "vector_rag":
        from retrieval.eval_run import run_vector_eval

        return run_vector_eval(eval_set=eval_set)
    if pipeline == "graph_rag":
        from graphrag.eval_run import run_eval

        return run_eval(eval_set=eval_set)
    from agentic.eval_run import run_agentic_eval

    return run_agentic_eval(eval_set=eval_set)


def main() -> None:
    open("eval_all_progress.log", "w").close()
    log(f"starting full evaluation suite — model: "
        f"{__import__('shared.llm_client', fromlist=['DEFAULT_MODEL']).DEFAULT_MODEL}")
    for eval_set in SETS:
        for pipeline in PIPELINES:
            log(f"=== START {pipeline} / {eval_set}")
            try:
                summary = run_one(pipeline, eval_set)
                log(f"=== DONE  {pipeline} / {eval_set}: {summary}")
            except Exception as exc:
                log(f"=== FAIL  {pipeline} / {eval_set}: {exc}")
                traceback.print_exc()
    log("ALL EVALUATIONS COMPLETE")


if __name__ == "__main__":
    main()
