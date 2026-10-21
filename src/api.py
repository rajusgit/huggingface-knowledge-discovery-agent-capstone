"""FastAPI app.

POST /api/chat streams server-sent events:
  {"type":"node","name":...}*  progress as each graph node finishes
  {"type":"token","text":...}* answer tokens
  {"type":"benchmark_status",...}* while approved benchmark jobs run
  {"type":"approval_required", kind, models, dataset, seed, limits}  stream ends; resume via POST /api/chat/resume
  {"type":"final", answer, routes, citations, invalid_citations, live_sources, tool_results, benchmarks, guard_events}
  {"type":"error","message":...} if the turn fails
GET /api/models/search, /api/datasets/search: filterable live Hugging Face discovery.
"""
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from agents import guards
from src import config
from tools import benchmarks, compare, hf_client

log = logging.getLogger("api")
_cfg = config.settings
SortKey = Literal["downloads", "likes", "lastModified", "createdAt", "trendingScore"]
_NS = uuid.UUID("6f1c1b7e-3b52-4d3e-9a63-0c7d3b0e2a11")  # thread ids are stored pseudonymously (uuid5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from agents import graph, mcp_tools, memory

    tools = await mcp_tools.load_tools()
    async with memory.postgres_checkpointer() as saver:
        app.state.agent = graph.build_agent(tools=tools, checkpointer=saver)
        yield


app = FastAPI(title="HF Knowledge & Discovery Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg.api.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    pipeline_variant: str = "default"
    memory: bool = False  # short-term memory is opt-in

    @field_validator("message")
    @classmethod
    def _message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        if len(v) > _cfg.api.max_message_chars:
            raise ValueError(f"message longer than {_cfg.api.max_message_chars} characters")
        return v


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def _persist(thread_id: str, question: str, final: dict, latency_ms: int) -> None:
    from src.db import get_pool

    tid = uuid.uuid5(_NS, thread_id)
    pool = get_pool()
    pool.open()
    with pool.connection() as conn:
        conn.execute("INSERT INTO sessions (thread_id, memory_enabled) VALUES (%s, true) ON CONFLICT DO NOTHING", (tid,))
        conn.execute("INSERT INTO messages (thread_id, role, content) VALUES (%s,'user',%s)", (tid, question))
        conn.execute(
            """INSERT INTO messages (thread_id, role, content, citations, llm_model, latency_ms)
               VALUES (%s,'assistant',%s,%s,%s,%s)""",
            (tid, final["answer"], json.dumps(final["citations"]), _cfg.llm["openrouter"]["model"], latency_ms),
        )


async def _events(stream, *, memory: bool, thread_id: str, question: str | None, pre_events: list[dict] | None = None):
    start = time.perf_counter()
    state: dict = {}
    streamed = awaiting = False
    try:
        async for mode, payload in stream:
            if mode == "interrupt":  # human approval needed: tell the client...
                awaiting = True
                yield _sse({"type": "approval_required", "kind": payload.get("type"),
                            **{k: v for k, v in payload.items() if k != "type"}})
                continue  # ...but keep draining: stopping here would cancel the checkpoint write
            if mode == "custom" and payload.get("type") in ("token", "benchmark_status"):
                streamed = streamed or payload["type"] == "token"
                yield _sse(payload)
            elif mode == "updates":
                for node, update in payload.items():
                    if node.startswith("__"):  # LangGraph-internal pseudo-nodes
                        continue
                    state.update(update or {})
                    yield _sse({"type": "node", "name": node})
        if awaiting:  # paused for approval; the client resumes via /api/chat/resume
            return
        answer = state.get("answer") or ""
        if not streamed and answer:  # refusals are produced without streaming
            yield _sse({"type": "token", "text": answer})
        final = {
            "type": "final",
            "answer": answer,
            "routes": state.get("routes", []),
            "citations": state.get("citations", []),
            "invalid_citations": state.get("invalid_citations", []),
            "live_sources": state.get("live_sources", []),
            "tool_results": state.get("live", []),
            "benchmarks": state.get("benchmarks", []),
            "guard_events": (pre_events or []) + state.get("guard_events", []),
        }
        # ponytail: resumed (approval) turns aren't written to `messages`; the question isn't known here
        if memory and question is not None:  # nothing is stored for memory-off turns
            try:
                await asyncio.to_thread(_persist, thread_id, state.get("question") or question, final,
                                        int((time.perf_counter() - start) * 1000))
            except Exception:
                log.exception("persisting chat turn failed")
        yield _sse(final)
    except Exception:
        log.exception("chat failed")
        yield _sse({"type": "error", "message": "Something went wrong generating the answer."})


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    # Mask PII *before* the graph runs: LangGraph checkpoints its input, so a later node would be too late.
    masked, pre_events = guards.mask_pii(req.message)
    stream = request.app.state.agent.astream(masked, req.thread_id, memory=req.memory)
    return StreamingResponse(
        _events(stream, memory=req.memory, thread_id=req.thread_id, question=masked, pre_events=pre_events),
        media_type="text/event-stream")


class ResumeRequest(BaseModel):
    thread_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    approved: bool
    memory: bool = False


@app.post("/api/chat/resume")
async def resume(req: ResumeRequest, request: Request) -> StreamingResponse:
    """Answer a pending human-approval request (e.g. run the proposed benchmark)."""
    stream = request.app.state.agent.aresume_stream(req.thread_id, req.approved, memory=req.memory)
    return StreamingResponse(
        _events(stream, memory=req.memory, thread_id=req.thread_id, question=None),
        media_type="text/event-stream")


def _discover(fn, *args, **kw) -> dict:
    try:
        return fn(*args, **kw)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except hf_client.HFError:
        log.exception("Hugging Face API call failed")
        raise HTTPException(502, "Hugging Face API is unavailable or rejected the request.")


@app.get("/api/models/search")
def search_models(q: str | None = None, task: str | None = None, library: str | None = None,
                  language: str | None = None, license: str | None = None, sort: SortKey = "downloads",
                  limit: int = Query(10, ge=1, le=_cfg.huggingface.max_results)) -> dict:
    return _discover(hf_client.search_models, q, task=task, library=library, language=language,
                     license=license, sort=sort, limit=limit)


@app.get("/api/datasets/search")
def search_datasets(q: str | None = None, language: str | None = None, license: str | None = None,
                    sort: SortKey = "downloads",
                    limit: int = Query(10, ge=1, le=_cfg.huggingface.max_results)) -> dict:
    return _discover(hf_client.search_datasets, q, language=language, license=license, sort=sort, limit=limit)


class CompareRequest(BaseModel):
    ids: list[str] = Field(max_length=compare.MAX_IDS)
    kind: Literal["models", "datasets"] = "models"
    constraints: dict | None = None


@app.post("/api/compare")
def compare_resources(req: CompareRequest) -> dict:
    return _discover(compare.compare_resources, req.ids, kind=req.kind, constraints=req.constraints)


def _start_in_background(job) -> None:
    benchmarks.start_in_background(job)


class BenchmarkRequest(BaseModel):
    model_ids: list[str] = Field(min_length=1, max_length=_cfg.benchmarks.max_models_per_benchmark)
    dataset_id: str
    seed: int | None = None


@app.post("/api/benchmarks")
def start_benchmarks(req: BenchmarkRequest) -> dict:
    """Start allowlisted benchmark jobs (human-initiated). Returns job ids to poll."""
    for m in req.model_ids:
        if m not in _cfg.benchmarks.allowed_models:
            raise HTTPException(422, f"model {m!r} is not on the benchmark allowlist")
    jobs = []
    try:
        for m in req.model_ids:
            jid = benchmarks.create_job(m, req.dataset_id, req.seed)
            _start_in_background(jid)
            jobs.append({"job_id": str(jid), "model_id": m, "status": "queued"})
    except benchmarks.BenchmarkRefused as e:
        raise HTTPException(422, str(e))
    return {"jobs": jobs}


@app.get("/api/benchmarks/{job_id}")
def get_benchmark(job_id: uuid.UUID) -> dict:
    rec = benchmarks.get_job(job_id)
    if rec is None:
        raise HTTPException(404, "benchmark job not found")
    return benchmarks.jsonable(rec)


@app.get("/api/evals/variants")
def eval_variants() -> dict:
    """The saved 16-variant leaderboard (written by standalone-scripts/run_evals.py)."""
    path = config.ROOT / config.settings.evals.results_file
    if not path.is_file():
        return {"variants": [], "winner": None, "message": "The variant experiment has not been run yet."}
    return json.loads(path.read_text())
