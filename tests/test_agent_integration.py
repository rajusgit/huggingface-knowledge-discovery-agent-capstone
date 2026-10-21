"""Real MCP stdio server + real Postgres checkpointer (fakes only for the LLM)."""
import asyncio
import uuid

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from agents import graph as g
from agents import mcp_tools, memory
from agents.router import Intent
from src import llm, retrieval

EXPECTED = {"search_models", "search_embedding_models", "get_model_details",
            "search_datasets", "get_dataset_details", "search_docs"}


def test_mcp_tools_load_and_convert_to_openai_format():
    tools = asyncio.run(asyncio.wait_for(mcp_tools.load_tools(), 120))
    assert EXPECTED <= {t.name for t in tools}
    for t in tools:
        spec = convert_to_openai_tool(t)
        assert spec["function"]["name"] == t.name and spec["function"]["parameters"]["type"] == "object"


def test_postgres_checkpointer_remembers_across_turns(monkeypatch):
    seen = []
    monkeypatch.setattr(retrieval, "search_knowledge", lambda *a, **k: [
        {"chunk_id": "c", "source_url": "https://d/x", "heading_path": ["T"], "revision": "r", "content": "text", "score": 0.9}])
    monkeypatch.setattr(llm, "stream", lambda m, **k: (seen.append(m), iter(["ok [1]"]))[1])

    class R:
        def classify(self, t):
            return Intent(["learn"])

    thread = f"test-{uuid.uuid4()}"

    async def go():
        try:
            async with memory.postgres_checkpointer() as saver:
                agent = g.build_agent(router=R(), tools=[], checkpointer=saver)
                try:
                    await agent.ainvoke("remember the word PINEAPPLE", thread)
                    await agent.ainvoke("what next?", thread)
                    await agent.ainvoke("fresh thread", f"{thread}-b")
                finally:
                    await saver.adelete_thread(thread)
                    await saver.adelete_thread(f"{thread}-b")
        except OSError as e:
            pytest.skip(f"database unavailable: {e}")

    try:
        asyncio.run(asyncio.wait_for(go(), 120))
    except Exception as e:
        if "connection" in str(e).lower() or "password" in str(e).lower():
            pytest.skip(f"database unavailable: {e}")
        raise
    assert any("PINEAPPLE" in m["content"] for m in seen[1])
    assert not any("PINEAPPLE" in m["content"] for m in seen[2])
