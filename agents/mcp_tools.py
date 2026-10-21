"""Load the project's FastMCP tools (local stdio server) as LangChain tools.

open_tools() keeps ONE server process for as long as the context is open. (client.get_tools() would start a
new process for every tool call; search_docs would then re-import torch and re-load the models each time.)
"""
import logging
import sys
from contextlib import asynccontextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from agents import tool_safety
from src import config

log = logging.getLogger("mcp_tools")

# Tools that must never be handed to the LLM's own tool loop: they need human approval first.
RESTRICTED = {"benchmark_embedding_model"}


@asynccontextmanager
async def open_tools(restricted: bool = False):
    client = MultiServerMCPClient({
        "hf": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", config.settings.agent.mcp_module],
            "cwd": str(config.ROOT),
        }
    })
    async with client.session("hf") as session:
        tools, rejected = tool_safety.filter_tools(await load_mcp_tools(session), set(config.settings.agent.allowed_tools))
        for name, why in rejected:
            log.warning("dropping MCP tool %s: %s", name, why)
        yield tools if restricted else [t for t in tools if t.name not in RESTRICTED]
