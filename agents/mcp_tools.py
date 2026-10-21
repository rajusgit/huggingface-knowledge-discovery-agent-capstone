"""Load the project's FastMCP tools (local stdio server) as LangChain tools."""
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

from src import config


async def load_tools() -> list:
    client = MultiServerMCPClient({
        "hf": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", config.settings.agent.mcp_module],
            "cwd": str(config.ROOT),
        }
    })
    return await client.get_tools()
