"""FastAPI app.

POST /api/chat streams server-sent events:
  {"type":"node","name":...}*  progress as each graph node finishes
  {"type":"token","text":...}* answer tokens
  {"type":"final", answer, routes, citations, invalid_citations, live_sources, tool_results, guard_events}
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

from src import config
from tools import hf_client

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


async def _chat_events(agent, req: ChatRequest):
    start = time.perf_counter()
    state: dict = {}
    streamed = False
    try:
        async for mode, payload in agent.astream(req.message, req.thread_id, memory=req.memory):
            if mode == "custom" and payload.get("type") == "token":
                streamed = True
                yield _sse(payload)
            elif mode == "updates":
                for node, update in payload.items():
                    state.update(update or {})
                    yield _sse({"type": "node", "name": node})
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
            "guard_events": [],
        }
        if req.memory:  # nothing is stored for memory-off turns
            try:
                await asyncio.to_thread(_persist, req.thread_id, req.message, final,
                                        int((time.perf_counter() - start) * 1000))
            except Exception:
                log.exception("persisting chat turn failed")
        yield _sse(final)
    except Exception:
        log.exception("chat failed")
        yield _sse({"type": "error", "message": "Something went wrong generating the answer."})


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    return StreamingResponse(_chat_events(request.app.state.agent, req), media_type="text/event-stream")


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
