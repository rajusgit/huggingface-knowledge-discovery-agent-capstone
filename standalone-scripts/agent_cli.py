"""Chat with the agent in the terminal:  uv run python standalone-scripts/agent_cli.py [--no-memory]
Questions are read from stdin (works interactively or piped)."""
import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import graph, mcp_tools, memory  # noqa: E402


async def main(use_memory: bool) -> None:
    async with mcp_tools.open_tools() as tools, memory.postgres_checkpointer() as saver:
        agent = graph.build_agent(tools=tools, checkpointer=saver)
        thread = str(uuid.uuid4())
        for line in sys.stdin:
            q = line.strip()
            if not q:
                continue
            print(f"\n> {q}")
            out = await agent.ainvoke(q, thread, memory=use_memory)
            print(f"[routes: {out['routes']}]\n{out['answer']}")
            for c in out["citations"]:
                print(f"  doc [{c['n']}] {' > '.join(c['heading_path'])} {c['source_url']}")
            for s in out["live_sources"][:6]:
                print(f"  live {s['tool']}: {s['source_url']} (fetched {s['fetched_at']})")
        await saver.adelete_thread(thread)


p = argparse.ArgumentParser()
p.add_argument("--no-memory", action="store_true")
asyncio.run(main(not p.parse_args().no_memory))
