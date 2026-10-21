import json

import pytest
from fastapi.testclient import TestClient

from src import api

CHUNKS = [{"chunk_id": "a", "source_url": "https://x/a", "heading_path": ["T", "A"], "revision": "r1",
           "content": "alpha", "score": 0.9}]


def events(resp):
    return [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ")]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api.rag, "prepare", lambda q, **k: ([{"role": "user", "content": q}], CHUNKS))
    monkeypatch.setattr(api.llm, "stream", lambda messages, **k: iter(["It is ", "alpha [1]", " (see [9])."]))
    return TestClient(api.app)


def post(client, **over):
    body = {"message": "what?", "thread_id": "t1", "pipeline_variant": "default", **over}
    return client.post("/api/chat", json=body)


def test_streams_tokens_then_final_with_citations(client):
    r = post(client)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = events(r)
    assert "".join(e["text"] for e in ev if e["type"] == "token") == "It is alpha [1] (see [9])."
    final = ev[-1]
    assert final["type"] == "final" and final["guard_events"] == []
    assert [c["chunk_id"] for c in final["citations"]] == ["a"] and final["citations"][0]["n"] == 1
    assert final["invalid_citations"] == [9]


def test_no_context_refuses_without_llm(monkeypatch):
    monkeypatch.setattr(api.rag, "prepare", lambda q, **k: ([], []))
    monkeypatch.setattr(api.llm, "stream", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM")))
    ev = events(post(TestClient(api.app)))
    assert "can't answer" in ev[0]["text"].lower() and ev[-1]["citations"] == []


def test_mid_stream_failure_becomes_error_event(monkeypatch):
    def broken(messages, **k):
        yield "partial"
        raise RuntimeError("boom")

    monkeypatch.setattr(api.rag, "prepare", lambda q, **k: ([], CHUNKS))
    monkeypatch.setattr(api.llm, "stream", broken)
    ev = events(post(TestClient(api.app)))
    assert ev[-1]["type"] == "error" and "boom" not in json.dumps(ev[-1])  # no internals leaked


@pytest.mark.parametrize("msg", ["", "   ", "x" * 5000])
def test_rejects_empty_or_oversized_message(client, msg):
    assert post(client, message=msg).status_code == 422


def test_cors_allows_configured_origin(client):
    r = client.options("/api/chat", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
