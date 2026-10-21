"""FastAPI app. POST /api/chat streams server-sent events:
   {"type":"token","text":...}* then {"type":"final","citations":[...],"invalid_citations":[...],"guard_events":[]}
   (or {"type":"error"} if generation fails mid-stream).
"""
import json
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from src import config, llm, prompts, rag

log = logging.getLogger("api")
app = FastAPI(title="HF Knowledge & Discovery Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.settings.api.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str
    pipeline_variant: str = "default"

    @field_validator("message")
    @classmethod
    def _message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        if len(v) > config.settings.api.max_message_chars:
            raise ValueError(f"message longer than {config.settings.api.max_message_chars} characters")
        return v


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _chat_events(req: ChatRequest):
    try:
        messages, chunks = rag.prepare(req.message)
        if not chunks:
            yield _sse({"type": "token", "text": prompts.NO_CONTEXT_ANSWER})
            yield _sse({"type": "final", "citations": [], "invalid_citations": [], "guard_events": []})
            return
        text = ""
        for tok in llm.stream(messages, temperature=0):
            text += tok
            yield _sse({"type": "token", "text": tok})
        ok, bad = prompts.extract_citations(text, len(chunks))
        yield _sse({
            "type": "final",
            "citations": [{**chunks[n - 1], "n": n} for n in ok],
            "invalid_citations": bad,
            "guard_events": [],
        })
    except Exception:
        log.exception("chat failed")
        yield _sse({"type": "error", "message": "Something went wrong generating the answer."})


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(_chat_events(req), media_type="text/event-stream")
