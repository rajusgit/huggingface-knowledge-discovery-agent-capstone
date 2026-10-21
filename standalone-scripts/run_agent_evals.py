"""Run the real agent (router, MCP tools, guards, LLM) over the golden set, one item at a time.

  uv run python standalone-scripts/run_agent_evals.py [--limit N]

Makes live LLM and Hugging Face calls. Benchmarks are only *proposed* (the run never approves one).
A tiny fixture dataset is made available for the duration of the run so the approval step can be exercised.
Writes observability/reports/agent_eval.json.
"""
import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import graph, mcp_tools  # noqa: E402
from observability import agent_eval, golden  # noqa: E402
from src import config  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, help="only the first N items per category")
a = p.parse_args()

items = golden.load_golden()
if a.limit:
    seen: dict[str, int] = {}
    items = [i for i in items if (seen.__setitem__(i["category"], seen.get(i["category"], 0) + 1) or seen[i["category"]] <= a.limit)]

dataset = config.ROOT / config.settings.benchmarks.dataset_dir / "bench_tiny.json"
provisioned = not dataset.exists()
if provisioned:
    dataset.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(config.ROOT / "tests" / "fixtures" / "bench_tiny.json", dataset)


async def main() -> dict:
    # one event loop and one MCP server process for the whole run (see agents/mcp_tools.py)
    async with mcp_tools.open_tools() as tools:
        print(f"{len(items)} items, tools: {[t.name for t in tools]}", flush=True)
        return await agent_eval.run_all(graph.build_agent(tools=tools), items)


try:
    result = asyncio.run(main())
finally:
    if provisioned:
        dataset.unlink(missing_ok=True)

out = config.ROOT / "observability" / "reports" / "agent_eval.json"
out.write_text(json.dumps(result, indent=1))
print(json.dumps({k: v for k, v in result.items() if k != "failures"}, indent=1))
print(f"{len(result['failures'])} items with a failed check; details in {out.relative_to(config.ROOT)}")
