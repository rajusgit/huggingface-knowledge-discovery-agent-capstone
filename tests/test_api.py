import json

import pytest
from fastapi.testclient import TestClient

from src import api
from tests.conftest import needs_db
from tools import hf_client

CITE = {"chunk_id": "a", "source_url": "https://x/a", "heading_path": ["T", "A"], "revision": "r1",
        "content": "alpha", "score": 0.9, "n": 1}
LIVE = [{"tool": "search_models", "args": {"query": "x"},
         "result": {"results": [{"id": "o/m1", "source_url": "https://huggingface.co/o/m1"}], "fetched_at": "2026-01-01T00:00:00Z"}}]

EVENTS = [
    ("updates", {"classify_intent": {"routes": ["learn", "discover"]}}),
    ("updates", {"retrieve_docs": {"chunks": [CITE]}}),
    ("updates", {"call_mcp_tools": {"live": LIVE, "live_sources": [{"tool": "search_models", "source_url": "https://huggingface.co/o/m1", "fetched_at": "2026-01-01T00:00:00Z"}]}}),
    ("custom", {"type": "token", "text": "It is "}),
    ("custom", {"type": "token", "text": "alpha [1]."}),
    ("updates", {"synthesize": {"answer": "It is alpha [1]."}}),
    ("updates", {"verify_citations": {"citations": [CITE], "invalid_citations": []}}),
    ("updates", {"respond": {"history": []}}),
]


class FakeAgent:
    def __init__(self, events=EVENTS):
        self.events, self.calls = events, []

    def astream(self, question, thread_id, memory=True, variant="default"):
        self.calls.append((question, thread_id, memory))

        async def gen():
            for e in self.events:
                if isinstance(e, Exception):
                    raise e
                yield e
        return gen()


def sse(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


@pytest.fixture
def agent(monkeypatch):
    a = FakeAgent()
    api.app.state.agent = a
    monkeypatch.setattr(api, "_persist", lambda *args, **kw: None)
    yield a
    del api.app.state.agent


@pytest.fixture
def client(agent):
    return TestClient(api.app)


def post(client, **over):
    return client.post("/api/chat", json={"message": "what?", "thread_id": "t1", **over})


def test_streams_progress_tokens_and_final(client):
    r = post(client)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = sse(r)
    assert [e["name"] for e in ev if e["type"] == "node"][:2] == ["classify_intent", "retrieve_docs"]
    assert "".join(e["text"] for e in ev if e["type"] == "token") == "It is alpha [1]."
    final = ev[-1]
    assert final["type"] == "final" and final["guard_events"] == [] and final["routes"] == ["learn", "discover"]
    assert final["citations"][0]["chunk_id"] == "a"
    assert final["live_sources"][0]["fetched_at"] == "2026-01-01T00:00:00Z"
    assert final["tool_results"][0]["result"]["fetched_at"] == "2026-01-01T00:00:00Z"
    assert final["answer"] == "It is alpha [1]."


def test_refusal_without_tokens_is_sent_as_a_token(agent, client):
    agent.events = [("updates", {"classify_intent": {"routes": []}}),
                    ("updates", {"validate_evidence": {"refused": True, "answer": "Out of scope."}})]
    ev = sse(post(client))
    assert [e["text"] for e in ev if e["type"] == "token"] == ["Out of scope."]
    assert ev[-1]["answer"] == "Out of scope."


def test_memory_defaults_off_and_flag_is_passed(agent, client):
    post(client)
    post(client, memory=True)
    assert [c[2] for c in agent.calls] == [False, True]


def test_failure_becomes_error_event_without_internals(agent, client):
    agent.events = [("custom", {"type": "token", "text": "partial"}), RuntimeError("secret boom")]
    ev = sse(post(client))
    assert ev[-1]["type"] == "error" and "secret boom" not in json.dumps(ev[-1])


@pytest.mark.parametrize("body", [{"message": ""}, {"message": "   "}, {"message": "x" * 5000},
                                  {"thread_id": "bad id!"}, {"thread_id": "x" * 100}])
def test_rejects_invalid_requests(client, body):
    assert post(client, **body).status_code == 422


def test_persists_only_when_memory_is_on(monkeypatch, agent):
    saved = []
    monkeypatch.setattr(api, "_persist", lambda *a, **k: saved.append(a))
    c = TestClient(api.app)
    post(c)
    assert saved == []
    post(c, memory=True)
    assert len(saved) == 1


def test_persist_failure_does_not_break_the_response(monkeypatch, agent):
    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(api, "_persist", boom)
    ev = sse(post(TestClient(api.app), memory=True))
    assert ev[-1]["type"] == "final"


def test_pii_is_masked_before_the_agent_and_in_storage(monkeypatch):
    class Spy(FakeAgent):
        pass
    spy = Spy(EVENTS[:3] + [("updates", {"guard_input": {"question": "mail me [EMAIL]", "guard_events": []}})] + EVENTS[3:])
    api.app.state.agent = spy
    saved = []
    monkeypatch.setattr(api, "_persist", lambda tid, q, final, ms: saved.append(q))
    try:
        ev = sse(TestClient(api.app).post("/api/chat", json={"message": "mail me jane@example.com", "thread_id": "t1", "memory": True}))
    finally:
        del api.app.state.agent
    assert spy.calls[0][0] == "mail me [EMAIL]"                       # the graph never sees the raw address
    assert saved == ["mail me [EMAIL]"]                               # nor does the database
    assert {"type": "pii_masked", "kind": "email", "count": 1} in ev[-1]["guard_events"]


def test_guard_events_from_the_graph_reach_the_client(agent, client):
    agent.events = [("updates", {"guard_input": {"question": "q", "guard_events": [{"type": "blocked", "reason": "potentially_harmful"}]}}),
                    ("updates", {"validate_evidence": {"answer": "I can't help with that request."}})]
    final = sse(post(client))[-1]
    assert final["guard_events"] == [{"type": "blocked", "reason": "potentially_harmful"}]


def test_cors_allows_configured_origin(client):
    r = client.options("/api/chat", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_model_search_maps_filters(monkeypatch, client):
    seen = {}
    monkeypatch.setattr(hf_client, "search_models", lambda q, **kw: seen.update(q=q, **kw) or {"results": [], "fetched_at": "t"})
    r = client.get("/api/models/search", params={"q": "multi", "task": "sentence-similarity", "library": "sentence-transformers",
                                                 "language": "fr", "license": "mit", "sort": "likes", "limit": 5})
    assert r.status_code == 200 and r.json()["fetched_at"] == "t"
    assert seen == {"q": "multi", "task": "sentence-similarity", "library": "sentence-transformers",
                    "language": "fr", "license": "mit", "sort": "likes", "limit": 5}


def test_dataset_search_maps_filters(monkeypatch, client):
    seen = {}
    monkeypatch.setattr(hf_client, "search_datasets", lambda q, **kw: seen.update(q=q, **kw) or {"results": [], "fetched_at": "t"})
    assert client.get("/api/datasets/search", params={"q": "squad", "language": "en", "license": "mit", "limit": 3}).status_code == 200
    assert seen["language"] == "en" and seen["license"] == "mit" and seen["limit"] == 3


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 9999}, {"sort": "DROP"}])
def test_search_validation_is_422(client, params):
    assert client.get("/api/models/search", params=params).status_code == 422


def test_upstream_failure_is_502_without_internals(monkeypatch, client):
    def boom(*a, **k):
        raise hf_client.HFError("Hugging Face API unreachable for /api/models")
    monkeypatch.setattr(hf_client, "search_models", boom)
    r = client.get("/api/models/search")
    assert r.status_code == 502


@needs_db
def test_persist_writes_session_and_messages_to_postgres():
    import uuid as _uuid

    import psycopg

    thread = f"test-{_uuid.uuid4()}"
    final = {"answer": "A [1]", "citations": [CITE]}
    try:
        api._persist(thread, "Q?", final, 12)
    except Exception as e:
        pytest.skip(f"database unavailable: {e}")
    tid = _uuid.uuid5(api._NS, thread)
    try:
        with psycopg.connect(api.config.settings.dsn, options="-c search_path=hf_discovery,public") as c:
            rows = c.execute("SELECT role, content, latency_ms FROM messages WHERE thread_id=%s ORDER BY message_id", (tid,)).fetchall()
            assert [r[0] for r in rows] == ["user", "assistant"] and rows[1][2] == 12
    finally:
        with psycopg.connect(api.config.settings.dsn, options="-c search_path=hf_discovery,public") as c:
            c.execute("DELETE FROM sessions WHERE thread_id=%s", (tid,))  # cascades to messages
