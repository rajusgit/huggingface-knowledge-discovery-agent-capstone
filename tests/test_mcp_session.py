import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from agents import mcp_tools


def test_open_tools_keeps_one_server_session_for_the_whole_lifetime(monkeypatch):
    """Regression: client.get_tools() starts a NEW server process for every tool call; for search_docs that meant
    re-loading torch and two models each time (20+ s)."""
    events = []

    class FakeClient:
        def __init__(self, conns, **kw):
            events.append(("init", list(conns)))

        @asynccontextmanager
        async def session(self, name):
            events.append(("enter", name))
            yield SimpleNamespace(name=name)
            events.append(("exit", name))

        async def get_tools(self, **kw):
            raise AssertionError("get_tools() opens a fresh session per call")

    async def fake_load(session, **kw):
        events.append(("load", session.name))
        mk = lambda n: SimpleNamespace(name=n, description="Search.", args_schema={"properties": {}})
        return [mk("search_models"), mk("benchmark_embedding_model"), mk("delete_everything")]

    monkeypatch.setattr(mcp_tools, "MultiServerMCPClient", FakeClient)
    monkeypatch.setattr(mcp_tools, "load_mcp_tools", fake_load)

    async def go():
        async with mcp_tools.open_tools() as tools:
            names = [t.name for t in tools]
            events.append(("using", names))
        async with mcp_tools.open_tools(restricted=True) as tools:
            return names, [t.name for t in tools]

    default, restricted = asyncio.run(go())
    assert default == ["search_models"]                                     # allowlist + restricted tool hidden
    assert restricted == ["search_models", "benchmark_embedding_model"]      # unknown tool always dropped
    kinds = [e[0] for e in events]
    assert kinds.index("exit") > kinds.index("using") and kinds.count("load") == 2
