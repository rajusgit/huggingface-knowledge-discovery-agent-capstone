import asyncio
import json
import uuid

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from agents import graph as g
from agents.router import Intent
from src import api, config, llm, prompts, retrieval
from tools import benchmarks

ALLOWED = config.settings.benchmarks.allowed_models
M1, M2 = ALLOWED[0], ALLOWED[1]


class R:
    def __init__(self, routes):
        self.routes = routes

    def classify(self, t):
        return Intent(self.routes)


def record(model, **kw):
    return {"job_id": str(uuid.uuid4()), "model_id": model, "model_revision": "rev1", "dataset_id": "bench_tiny",
            "dataset_revision": "fx1", "status": "succeeded", "seed": 0, "hardware": {"cpu_count": 8},
            "metrics": {"recall@5": 0.9, "mrr@10": 0.8, "ndcg@10": 0.85, "timing": {"docs_per_second": 100.0},
                        "n_docs": 12, "n_queries": 6}, "error": None, **kw}


@pytest.fixture
def bench(monkeypatch):
    calls = {"create": [], "run": []}
    recs = {}

    def create(model_id, dataset_id, seed=None):
        jid = uuid.uuid4()
        calls["create"].append((model_id, dataset_id, seed))
        recs[jid] = record(model_id, job_id=str(jid))
        return jid

    monkeypatch.setattr(benchmarks, "list_datasets", lambda: ["bench_tiny"])
    monkeypatch.setattr(benchmarks, "create_job", create)
    monkeypatch.setattr(benchmarks, "start_in_background", lambda jid: calls["run"].append(jid))
    monkeypatch.setattr(benchmarks, "get_job", lambda jid: recs[uuid.UUID(str(jid))])
    monkeypatch.setattr(g.config.settings.benchmarks, "poll_interval_seconds", 0)
    monkeypatch.setattr(retrieval, "search_knowledge", lambda *a, **k: [])
    return calls


def llm_script(monkeypatch, proposal, answer="Measured results [none]."):
    seen = {"stream": []}
    monkeypatch.setattr(llm, "complete", lambda m, **k: {"content": json.dumps(proposal), "tool_calls": []})
    monkeypatch.setattr(llm, "stream", lambda m, **k: (seen["stream"].append(m), iter([answer]))[1])
    return seen


def agent_with(saver=None):
    saver = saver or InMemorySaver()
    return g.build_agent(router=R(["evaluate"]), tools=[], checkpointer=saver), saver


def test_evaluate_pauses_for_approval_and_runs_nothing_first(bench, monkeypatch):
    llm_script(monkeypatch, {"models": [M1, M2], "dataset": "bench_tiny"})
    agent, _ = agent_with()
    out = asyncio.run(agent.ainvoke("benchmark the two models", "T"))
    assert out["interrupt"]["type"] == "benchmark_approval"
    assert out["interrupt"]["models"] == [M1, M2] and out["interrupt"]["dataset"] == "bench_tiny"
    assert out["interrupt"]["limits"]["max_runtime_seconds"] == config.settings.benchmarks.max_runtime_seconds
    assert bench["create"] == [] and bench["run"] == []          # nothing ran


def test_approval_runs_jobs_and_returns_measured_results(bench, monkeypatch):
    seen = llm_script(monkeypatch, {"models": [M1, M2], "dataset": "bench_tiny"})
    agent, _ = agent_with()
    asyncio.run(agent.ainvoke("benchmark both", "T"))
    out = asyncio.run(agent.aresume("T", approved=True))
    assert [c[:2] for c in bench["create"]] == [(M1, "bench_tiny"), (M2, "bench_tiny")] and len(bench["run"]) == 2
    assert out["interrupt"] is None and len(out["benchmarks"]) == 2
    assert out["benchmarks"][0]["metrics"]["recall@5"] == 0.9 and out["benchmarks"][0]["model_revision"] == "rev1"
    synth = seen["stream"][0]
    assert "MEASURED" in synth[0]["content"] and "bench_tiny" in synth[-1]["content"]


def test_rejection_ends_cleanly_without_running(bench, monkeypatch):
    llm_script(monkeypatch, {"models": [M1], "dataset": "bench_tiny"})
    agent, _ = agent_with()
    asyncio.run(agent.ainvoke("benchmark", "T"))
    out = asyncio.run(agent.aresume("T", approved=False))
    assert out["interrupt"] is None and "not approved" in out["answer"].lower()
    assert bench["create"] == [] and bench["run"] == []


def test_llm_proposed_parameters_are_validated_against_allowlists(bench, monkeypatch):
    llm_script(monkeypatch, {"models": ["evil/backdoor", M1], "dataset": "bench_tiny"})
    agent, _ = agent_with()
    out = asyncio.run(agent.ainvoke("benchmark evil/backdoor and the first", "T"))
    assert out["interrupt"]["models"] == [M1]                    # unknown model silently dropped


def test_nothing_valid_asks_for_clarification_without_interrupt(bench, monkeypatch):
    llm_script(monkeypatch, {"models": ["evil/backdoor"], "dataset": "../../etc"})
    agent, _ = agent_with()
    out = asyncio.run(agent.ainvoke("benchmark evil", "T"))
    assert out["interrupt"] is None and M1 in out["answer"] and "bench_tiny" in out["answer"]
    assert bench["create"] == []


def test_no_datasets_available_is_explained(bench, monkeypatch):
    monkeypatch.setattr(benchmarks, "list_datasets", lambda: [])
    llm_script(monkeypatch, {"models": [M1], "dataset": "x"})
    agent, _ = agent_with()
    out = asyncio.run(agent.ainvoke("benchmark", "T"))
    assert out["interrupt"] is None and "no benchmark datasets" in out["answer"].lower()


def test_failed_job_is_reported_not_hidden(bench, monkeypatch):
    llm_script(monkeypatch, {"models": [M1], "dataset": "bench_tiny"})
    monkeypatch.setattr(benchmarks, "get_job", lambda jid: record(M1, status="timeout", metrics=None, error="exceeded the 600s runtime limit"))
    agent, _ = agent_with()
    asyncio.run(agent.ainvoke("benchmark", "T"))
    out = asyncio.run(agent.aresume("T", approved=True))
    assert out["benchmarks"][0]["status"] == "timeout" and "runtime limit" in out["benchmarks"][0]["error"]


def test_memory_off_deletes_thread_state_when_finished(bench, monkeypatch):
    saver = InMemorySaver()
    agent, _ = agent_with(saver)
    llm_script(monkeypatch, {"models": [M1], "dataset": "bench_tiny"})
    asyncio.run(agent.ainvoke("benchmark", "T", memory=False))
    private = {"configurable": {"thread_id": "T~nomem"}}
    assert saver.get_tuple(private) is not None            # pending approval keeps its state
    asyncio.run(agent.aresume("T", approved=True, memory=False))
    assert saver.get_tuple(private) is None                # cleaned up once the turn finishes


def test_memory_off_normal_turn_leaves_nothing_and_ignores_history(monkeypatch):
    saver = InMemorySaver()
    seen = {"s": []}
    monkeypatch.setattr(retrieval, "search_knowledge", lambda *a, **k: [{"chunk_id": "c", "source_url": "u", "heading_path": ["T"], "revision": "r", "content": "x", "score": 1}])
    monkeypatch.setattr(llm, "stream", lambda m, **k: (seen["s"].append(m), iter(["ok [1]"]))[1])
    agent = g.build_agent(router=R(["learn"]), tools=[], checkpointer=saver)
    asyncio.run(agent.ainvoke("secret first question", "T", memory=True))
    asyncio.run(agent.ainvoke("second", "T", memory=False))
    assert not any("secret first question" in m["content"] for m in seen["s"][1])
    assert saver.get_tuple({"configurable": {"thread_id": "T"}}) is not None      # memory-on state untouched
    assert saver.get_tuple({"configurable": {"thread_id": "T~nomem"}}) is None    # off turn left nothing


# ---- API ------------------------------------------------------------------------

class FakeAgent:
    def __init__(self, events):
        self.events, self.calls = events, []

    def _gen(self):
        async def gen():
            for e in self.events:
                yield e
        return gen()

    def astream(self, q, t, memory=True, variant="default"):
        self.calls.append(("stream", q, t, memory))
        return self._gen()

    def aresume_stream(self, t, approved, memory=True):
        self.calls.append(("resume", t, approved, memory))
        return self._gen()


def sse(resp):
    return [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ")]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_persist", lambda *a, **k: None)
    yield TestClient(api.app)
    if hasattr(api.app.state, "agent"):
        del api.app.state.agent


APPROVAL = {"type": "benchmark_approval", "models": [M1], "dataset": "bench_tiny", "seed": 0,
            "limits": {"max_runtime_seconds": 600, "max_memory_mb": 4096}}


def test_api_chat_surfaces_approval_request_and_ends_without_final(client):
    api.app.state.agent = FakeAgent([("updates", {"classify_intent": {"routes": ["evaluate"]}}), ("interrupt", APPROVAL)])
    ev = sse(client.post("/api/chat", json={"message": "benchmark", "thread_id": "t1"}))
    assert ev[-1]["type"] == "approval_required" and ev[-1]["models"] == [M1]
    assert not any(e["type"] == "final" for e in ev)


def test_api_resume_streams_status_and_final_with_benchmarks(client):
    rec = record(M1)
    agent = FakeAgent([("custom", {"type": "benchmark_status", "jobs": [{"job_id": rec["job_id"], "status": "running"}]}),
                       ("updates", {"poll_results": {"benchmarks": [rec], "live": []}}),
                       ("updates", {"synthesize": {"answer": "Done."}})])
    api.app.state.agent = agent
    r = client.post("/api/chat/resume", json={"thread_id": "t1", "approved": True, "memory": False})
    ev = sse(r)
    assert any(e["type"] == "benchmark_status" for e in ev)
    assert ev[-1]["type"] == "final" and ev[-1]["benchmarks"][0]["metrics"]["recall@5"] == 0.9
    assert agent.calls == [("resume", "t1", True, False)]


def test_api_start_benchmark_creates_and_runs_allowlisted_jobs(monkeypatch, client):
    started = []
    monkeypatch.setattr(benchmarks, "create_job", lambda m, d, s=None: uuid.UUID(int=len(started) + 1))
    monkeypatch.setattr(api, "_start_in_background", lambda jid: started.append(jid))
    r = client.post("/api/benchmarks", json={"model_ids": [M1, M2], "dataset_id": "bench_tiny", "seed": 3})
    assert r.status_code == 200 and len(r.json()["jobs"]) == 2 and len(started) == 2
    assert r.json()["jobs"][0]["status"] == "queued"


def test_api_start_benchmark_rejects_refused_requests(monkeypatch, client):
    def refuse(m, d, s=None):
        raise benchmarks.BenchmarkRefused("model is not on the benchmark allowlist")
    monkeypatch.setattr(benchmarks, "create_job", refuse)
    assert client.post("/api/benchmarks", json={"model_ids": ["evil/x"], "dataset_id": "bench_tiny"}).status_code == 422
    assert client.post("/api/benchmarks", json={"model_ids": [], "dataset_id": "bench_tiny"}).status_code == 422
    assert client.post("/api/benchmarks", json={"model_ids": [M1] * 9, "dataset_id": "bench_tiny"}).status_code == 422


def test_api_get_benchmark(monkeypatch, client):
    rec = record(M1)
    monkeypatch.setattr(benchmarks, "get_job", lambda j: rec if str(j) == rec["job_id"] else None)
    assert client.get(f"/api/benchmarks/{rec['job_id']}").json()["metrics"]["recall@5"] == 0.9
    assert client.get(f"/api/benchmarks/{uuid.uuid4()}").status_code == 404
    assert client.get("/api/benchmarks/not-a-uuid").status_code == 422


def test_api_drains_stream_after_interrupt_so_checkpoint_is_saved(client):
    """Stopping early cancels LangGraph before it persists the pending approval."""
    done = []

    class Agent2(FakeAgent):
        def astream(self, q, t, memory=True, variant="default"):
            async def gen():
                yield ("interrupt", APPROVAL)
                yield ("updates", {"__metadata__": {"cached": True}})
                done.append("persisted")          # runs only if the consumer keeps reading
            return gen()

    api.app.state.agent = Agent2([])
    ev = sse(client.post("/api/chat", json={"message": "benchmark", "thread_id": "t1"}))
    assert done == ["persisted"]
    assert [e["type"] for e in ev] == ["approval_required"]       # internal pseudo-nodes hidden, no final
