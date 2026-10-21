"""Load the project's FastMCP tools (local stdio server) as LangChain tools."""
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

from src import config


# Tools that must never be handed to the LLM's own tool loop: they need human approval first.
RESTRICTED = {"benchmark_embedding_model"}


async def load_tools(restricted: bool = False) -> list:
    client = MultiServerMCPClient({
        "hf": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", config.settings.agent.mcp_module],
            "cwd": str(config.ROOT),
        }
    })
    tools = await client.get_tools()
    return tools if restricted else [t for t in tools if t.name not in RESTRICTED]
