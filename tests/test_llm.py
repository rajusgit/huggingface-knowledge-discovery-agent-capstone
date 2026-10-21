import ast
import sys
from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest

from src import llm
from src.config import ROOT


def resp(content="hi", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 3, "completion_tokens": 2})
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage, model="m")


class Fake:
    """Stand-in for openai.OpenAI; `script` items are responses or exceptions."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        self.last = kw
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def rate_limit():
    req = httpx.Request("POST", "http://x")
    return openai.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(llm, "_sleep", lambda s: None)
    llm.reset()


def install(monkeypatch, script):
    fake = Fake(script)
    monkeypatch.setattr(llm, "_client", fake)
    return fake


def test_complete_returns_plain_dict(monkeypatch):
    install(monkeypatch, [resp("hello")])
    out = llm.complete([{"role": "user", "content": "x"}])
    assert out["content"] == "hello" and out["usage"]["prompt_tokens"] == 3
    assert out["model"] == "m" and out["tool_calls"] == [] and out["latency_ms"] >= 0


def test_tool_calls_are_plain_dicts(monkeypatch):
    tc = SimpleNamespace(id="1", function=SimpleNamespace(name="f", arguments="{}"))
    install(monkeypatch, [resp(None, [tc])])
    out = llm.complete([], tools=[{"type": "function"}])
    assert out["tool_calls"] == [{"id": "1", "name": "f", "arguments": "{}"}]


def test_three_retries_then_raises(monkeypatch):
    fake = install(monkeypatch, [rate_limit()] * 4)
    with pytest.raises(llm.LLMError):
        llm.complete([{"role": "user", "content": "x"}])
    assert fake.calls == 4  # 1 attempt + 3 retries


def test_retry_then_success(monkeypatch):
    fake = install(monkeypatch, [rate_limit(), resp("ok")])
    assert llm.complete([])["content"] == "ok" and fake.calls == 2


def test_non_retryable_error_not_retried(monkeypatch):
    req = httpx.Request("POST", "http://x")
    bad = openai.AuthenticationError("no", response=httpx.Response(401, request=req), body=None)
    fake = install(monkeypatch, [bad])
    with pytest.raises(llm.LLMError):
        llm.complete([])
    assert fake.calls == 1


def test_missing_key_raises_before_network(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setattr(llm, "_load_dotenv", lambda: None)
    with pytest.raises(llm.LLMConfigError, match="OPENROUTER_API_KEY"):
        llm.complete([])


def test_temperature_zero_is_cached(monkeypatch):
    fake = install(monkeypatch, [resp("a"), resp("b")])
    m = [{"role": "user", "content": "x"}]
    assert llm.complete(m, temperature=0)["content"] == "a"
    assert llm.complete(m, temperature=0)["content"] == "a"
    assert fake.calls == 1


def test_not_cached_without_temperature_zero(monkeypatch):
    fake = install(monkeypatch, [resp("a"), resp("b")])
    m = [{"role": "user", "content": "x"}]
    llm.complete(m)
    llm.complete(m)
    assert fake.calls == 2


def test_stream_yields_text(monkeypatch):
    def chunk(t):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))])
    install(monkeypatch, [iter([chunk("a"), chunk(None), chunk("b")])])
    assert "".join(llm.stream([])) == "ab"


def test_llm_module_imports_nothing_from_src():
    tree = ast.parse((ROOT / "src" / "llm.py").read_text())
    allowed = set(sys.stdlib_module_names) | {"yaml", "openai"}
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        for n in names:
            assert n in allowed, f"src/llm.py must not import {n}"


def test_single_gateway_no_other_module_imports_openai():
    for pkg in ("src", "agents", "tools", "observability"):
        for path in (ROOT / pkg).rglob("*.py"):
            if path.name == "llm.py" and pkg == "src":
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    mods = [node.module or ""]
                assert not any(m.split(".")[0] == "openai" for m in mods), f"{path} imports openai"
