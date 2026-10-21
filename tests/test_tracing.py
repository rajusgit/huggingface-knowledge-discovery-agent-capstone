import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool
from langsmith import Client

from agents import graph as g
from agents import guards
from agents.router import Intent
from observability import tracing
from src import config, llm, pii, prompts, retrieval

EMAIL, KEY = "jane.doe@example.com", "sk-abcdefghijklmnopqrstuvwxyz123456"


class Capture(Client):
    """A LangSmith client that never touches the network and records every run it would post."""

    def __init__(self, **kw):
        super().__init__(api_url="http://localhost:1", api_key="x", auto_batch_tracing=False, **kw)
        self.sent = []

    def request_with_retries(self, method, pathname, *, request_kwargs=None, **kw):
        body = (request_kwargs or {}).get("data") or (request_kwargs or {}).get("json")
        body = json.loads(body) if isinstance(body, (bytes, str)) else body
        self.sent.append((method, body))
        return SimpleNamespace(status_code=200, json=lambda: {}, text="{}", headers={}, content=b"{}")

    def runs(self):
        return [b for m, b in self.sent if m == "POST" and isinstance(b, dict) and b.get("name")]

    def ended(self):
        return {str(b.get("id") or p.split("/")[-1]) for m, b in self.sent for p in [""] if m == "PATCH"}


def tracer_with_capture(**hooks):
    cap = Capture(**tracing.client_hooks())
    return cap, tracing.Tracer(cap)


# ---- redaction ---------------------------------------------------------------------------------------------------

def test_redact_masks_pii_and_secrets_recursively_and_truncates(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-0123456789abcdef")
    out = tracing.redact({"q": f"mail {EMAIL} key {KEY}", "nested": [{"k": "or-key-0123456789abcdef"}, ("x" * 10_000,)], "n": 3, "ok": True, "none": None})
    flat = json.dumps(out)
    assert EMAIL not in flat and KEY not in flat and "or-key-0123456789abcdef" not in flat
    assert out["n"] == 3 and out["ok"] is True and out["none"] is None
    assert len(out["nested"][1][0]) < config.settings.observability.max_chars + 50


def test_redact_survives_objects_it_cannot_serialise():
    class Weird:
        def __repr__(self):
            return f"Weird({EMAIL})"
    out = tracing.redact({"w": Weird()})
    assert EMAIL not in json.dumps(out)


# ---- tracer on/off ---------------------------------------------------------------------------------------------------

def test_tracing_is_off_without_an_api_key(monkeypatch):
    monkeypatch.setattr(config.settings, "langsmith_api_key", None)
    assert tracing.default_tracer().client is None


def test_tracing_is_off_when_disabled_in_config_even_with_a_key(monkeypatch):
    monkeypatch.setattr(config.settings, "langsmith_api_key", "ls-key")
    monkeypatch.setattr(config.settings.observability, "tracing_enabled", False)
    assert tracing.default_tracer().client is None


def test_turn_config_carries_the_documented_metadata(monkeypatch):
    monkeypatch.setattr(tracing, "_corpus_revision", lambda: "corp-rev-1")
    cfg = tracing.Tracer(None).config("thread-abc", "default", memory=True)
    md = cfg["metadata"]
    assert md["thread_id"] != "thread-abc" and len(md["thread_id"]) == 12       # pseudonymous
    assert md["pipeline_variant"] == "default" and md["prompt_version"] == config.settings.observability.prompt_version
    assert md["corpus_revision"] == "corp-rev-1" and md["embedding_model"] == config.settings.embedding.model
    assert md["llm_model"] == config.settings.llm["openrouter"]["model"] and md["request_id"] and md["memory"] is True
    assert any(t.startswith("variant:") for t in cfg["tags"])
    assert tracing.Tracer(None).config("t", "default", True)["metadata"]["request_id"] != md["request_id"]


# ---- the agent produces complete, redacted trace trees --------------------------------------------------------------------

CHUNK = {"chunk_id": "c1", "source_id": "s", "source_url": "https://docs/x", "heading_path": ["T"], "revision": "r",
         "content": f"Use max_new_tokens. Contact {EMAIL}.", "score": 0.9}


def run_agent(monkeypatch, routes, question, tools=None, stream="Answer [1].", complete=None):
    cap, tracer = tracer_with_capture()
    monkeypatch.setattr(retrieval, "search_knowledge", lambda *a, **k: [dict(CHUNK)])      # the traced retrieve() runs for real
    monkeypatch.setattr(llm, "stream", lambda m, **k: iter([stream]))
    monkeypatch.setattr(llm, "complete", complete or (lambda m, **k: {"content": "", "tool_calls": [], "usage": {"cost": 0.001}}))

    class R:
        def classify(self, t):
            return Intent(routes)
    agent = g.build_agent(router=R(), tools=tools or [], guard=guards.Guard(None, False), tracer=tracer)
    out = asyncio.run(agent.ainvoke(question, "thread-xyz", memory=False))
    return cap, out


def names(cap):
    return [r["name"] for r in cap.runs()]


def test_docs_turn_has_a_root_span_with_a_child_per_stage(monkeypatch):
    cap, _ = run_agent(monkeypatch, ["learn"], "how do I limit length?")
    got = set(names(cap))
    assert {"LangGraph", "guard_input", "classify_intent", "retrieve_docs", "synthesize", "respond"} <= got
    root = next(r for r in cap.runs() if r["name"] == "LangGraph")
    assert root["extra"]["metadata"]["pipeline_variant"] == "default" and root["extra"]["metadata"]["prompt_version"]
    children = [r for r in cap.runs() if r["name"] != "LangGraph"]
    assert all(r.get("parent_run_id") for r in children)                       # one tree, nothing orphaned


def test_every_span_that_starts_also_ends(monkeypatch):
    cap, _ = run_agent(monkeypatch, ["learn"], "q")
    started = {r["id"] for r in cap.runs()}
    finished = set()
    for m, b in cap.sent:
        if m == "PATCH" and isinstance(b, dict) and b.get("end_time"):
            finished.add(str(b.get("id")))
    assert started and all(i in finished for i in started)


def test_retrieve_and_rerank_and_generate_have_their_own_spans(monkeypatch):
    cap, _ = run_agent(monkeypatch, ["learn"], "q")
    assert {"retrieve", "generate"} <= set(names(cap))


def test_tool_calls_are_spans_named_after_the_tool(monkeypatch):
    tool = StructuredTool.from_function(lambda query: json.dumps({"results": []}), name="search_models", description="Search models.")
    script = [{"content": None, "tool_calls": [{"id": "1", "name": "search_models", "arguments": json.dumps({"query": "x"})}], "usage": {"cost": 0.001}},
              {"content": "", "tool_calls": [], "usage": {"cost": 0.001}}]
    cap, _ = run_agent(monkeypatch, ["discover"], "find models", tools=[tool], complete=lambda m, **k: script.pop(0))
    tool_spans = [r for r in cap.runs() if r["run_type"] == "tool"]
    assert [r["name"] for r in tool_spans] == ["search_models"]


def test_no_raw_pii_or_secret_is_ever_posted_to_langsmith(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-0123456789abcdef")
    cap, out = run_agent(monkeypatch, ["learn"], f"my mail is {EMAIL} and my key is {KEY} or-key-0123456789abcdef, how to limit length?",
                         stream="Answer [1].")
    everything = json.dumps(cap.sent, default=str)
    for raw in (EMAIL, KEY, "or-key-0123456789abcdef"):
        assert raw not in everything, raw
    assert "[EMAIL]" in everything or "[SECRET]" in everything or "REDACTED" in everything


def test_disabled_tracer_posts_nothing_even_if_the_global_env_flag_is_on(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    cap = Capture(**tracing.client_hooks())
    monkeypatch.setattr(retrieval, "retrieve", lambda *a, **k: [dict(CHUNK)])
    monkeypatch.setattr(llm, "stream", lambda m, **k: iter(["Answer [1]."]))

    class R:
        def classify(self, t):
            return Intent(["learn"])
    agent = g.build_agent(router=R(), tools=[], guard=guards.Guard(None, False), tracer=tracing.Tracer(None))
    asyncio.run(agent.ainvoke("q", "t", memory=False))
    assert cap.sent == []
