import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.config import ROOT

EXPECTED = {"search_models", "search_embedding_models", "get_model_details",
            "search_datasets", "get_dataset_details", "search_docs", "get_card_section",
            "compare_resources", "benchmark_embedding_model"}


async def _list():
    params = StdioServerParameters(command=sys.executable, args=["-m", "tools.mcp_server"], cwd=str(ROOT))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            return (await s.list_tools()).tools


def test_server_lists_all_tools_over_stdio():
    tools = asyncio.run(asyncio.wait_for(_list(), timeout=120))
    assert EXPECTED <= {t.name for t in tools}
    for t in tools:
        assert t.description and "ignore" not in t.description.lower()
