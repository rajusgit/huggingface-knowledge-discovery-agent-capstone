"""Run the 16-variant RAG experiment on the golden set.

  uv run python standalone-scripts/run_evals.py [--limit N] [--no-judges] [--apply-winner]

Needs an ingest run per chunker over the active corpus (ingest.py --all, then --strategy for the
other three). Writes evals.results_file (JSON) and evals.report_file (markdown).
--limit N keeps only the first N items per category (cheap smoke run).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg  # noqa: E402

from observability import evals, golden  # noqa: E402
from src import config  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int)
p.add_argument("--no-judges", action="store_true", help="deterministic metrics only (no LLM judge calls)")
p.add_argument("--apply-winner", action="store_true", help="write the winner into config.yaml (retrieval.*)")
a = p.parse_args()

items = golden.load_golden()
if a.limit:
    seen: dict[str, int] = {}
    keep = []
    for i in items:  # first N per category, but keep both splits represented (every 4th is heldout)
        seen[i["category"]] = seen.get(i["category"], 0) + 1
        if seen[i["category"]] <= a.limit:
            keep.append(i)
    items = keep
if a.no_judges:
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        setattr(evals, name, lambda o, e: None)

with psycopg.connect(config.settings.dsn, options=f"-c search_path={config.settings.database.schema_name},public") as conn:
    runs = evals.resolve_runs(conn)
print("runs per chunker:", runs, f"| {len(items)} golden items")
res = evals.run_experiment(items, runs)

out = config.ROOT / config.settings.evals.results_file
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=1))
(config.ROOT / config.settings.evals.report_file).write_text(evals.leaderboard_markdown(res))
print((config.ROOT / config.settings.evals.report_file).read_text())
if a.apply_winner:
    w = res["winner"]["variant"].split("|")
    evals.apply_winner(evals.Variant(w[0], w[1], w[2] == "rerank"), res["experiment_id"])
    print("config.yaml retrieval.* now points at", res["winner"]["variant"])
