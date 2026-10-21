"""LangGraph agent (hand-built nodes; every LLM call goes through src.llm).

START -> classify_intent -> [retrieve_docs] -> [call_mcp_tools] -> validate_evidence
      -> synthesize -> verify_citations -> respond -> END

Tool outputs and retrieved text are untrusted data: delimited, never executed as instructions.
Memory = `history` in state, persisted by the checkpointer keyed by thread_id.
"""
import asyncio
import json
import operator
from typing import Annotated, TypedDict

from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from agents import router as router_mod
from src import config, llm, prompts, retrieval


class State(TypedDict, total=False):
    question: str
    history: Annotated[list[dict], operator.add]
    routes: list[str]
    chunks: list[dict]
    live: list[dict]
    refused: bool
    answer: str
    citations: list[dict]
    invalid_citations: list[int]
    live_sources: list[dict]


def _live_sources(tool: str, result) -> list[dict]:
    """Collect {tool, source_url, fetched_at} from a tool's JSON result (provenance for the UI)."""
    out = []
    fetched = result.get("fetched_at") if isinstance(result, dict) else None

    def walk(x):
        if isinstance(x, dict):
            if x.get("source_url"):
                out.append({"tool": tool, "source_url": x["source_url"], "fetched_at": x.get("fetched_at", fetched)})
            for v in x.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(result)
    return out


def _tool_text(raw) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):  # MCP content blocks
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    return json.dumps(raw, default=str)


def build_graph(router, tools, checkpointer=None):
    cfg = config.settings.agent
    tools = tools or []
    by_name = {t.name: t for t in tools}
    openai_tools = [convert_to_openai_tool(t) for t in tools]

    def classify_intent(state: State):
        intent = router.classify(state["question"])
        return {"routes": intent.routes, "chunks": [], "live": [], "refused": False, "answer": "",
                "citations": [], "invalid_citations": [], "live_sources": []}

    def after_classify(state: State) -> str:
        docs, mcp = router_mod.branches(state["routes"])
        return "retrieve_docs" if docs else "call_mcp_tools" if mcp else "validate_evidence"

    def after_docs(state: State) -> str:
        return "call_mcp_tools" if router_mod.branches(state["routes"])[1] else "validate_evidence"

    def retrieve_docs(state: State):
        hits = retrieval.search_knowledge(state["question"])
        return {"chunks": [c for c in hits if c.get("content")]}

    async def call_mcp_tools(state: State):
        if not by_name:
            return {"live": []}
        history = state.get("history", [])[-2 * cfg.history_turns:]
        messages = [{"role": "system", "content": prompts.SYSTEM_TOOLS}, *history,
                    {"role": "user", "content": state["question"]}]
        live: list[dict] = []
        for _ in range(cfg.max_tool_rounds):
            out = await asyncio.to_thread(llm.complete, messages, tools=openai_tools)
            if not out["tool_calls"]:
                break
            messages.append({"role": "assistant", "content": out["content"], "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in out["tool_calls"]]})
            for call in out["tool_calls"]:
                result, text = None, None
                tool = by_name.get(call["name"])
                try:
                    if tool is None:
                        raise ValueError(f"tool {call['name']!r} is not allowed")
                    args = json.loads(call["arguments"] or "{}")
                    text = _tool_text(await tool.ainvoke(args))
                    try:
                        result = json.loads(text)
                    except ValueError:
                        result = text
                    live.append({"tool": call["name"], "args": args, "result": result})
                except Exception as e:  # report to the model; never crash the turn
                    text = json.dumps({"error": f"tool call failed: {type(e).__name__}: {e}"})
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": prompts.wrap_untrusted(text[:cfg.max_tool_result_chars])})
        return {"live": live, "live_sources": [s for l in live for s in _live_sources(l["tool"], l["result"])]}

    def validate_evidence(state: State):
        if not state["routes"]:
            return {"refused": True, "answer": prompts.OUT_OF_SCOPE_ANSWER}
        if not state["chunks"] and not state["live"]:
            return {"refused": True, "answer": prompts.NO_CONTEXT_ANSWER}
        return {}

    def synthesize(state: State):
        if state["refused"]:
            return {}
        parts = []
        if state["chunks"]:
            parts.append("Documentation passages:\n" + prompts.build_context(state["chunks"]))
        if state["live"]:
            data = json.dumps([{"tool": l["tool"], "args": l["args"], "result": l["result"]} for l in state["live"]],
                              default=str)[:cfg.max_tool_result_chars * 2]
            parts.append("Live Hugging Face data:\n" + prompts.wrap_untrusted(data))
        history = state.get("history", [])[-2 * cfg.history_turns:]
        messages = [{"role": "system", "content": prompts.SYSTEM_AGENT}, *history,
                    {"role": "user", "content": "\n\n".join(parts) + f"\n\nQuestion: {state['question']}"}]
        writer = get_stream_writer()
        text = ""
        for tok in llm.stream(messages, temperature=0):
            text += tok
            writer({"type": "token", "text": tok})
        return {"answer": text.strip()}

    def verify_citations(state: State):
        if state["refused"]:
            return {}
        ok, bad = prompts.extract_citations(state["answer"], len(state["chunks"]))
        return {"citations": [{**state["chunks"][n - 1], "n": n} for n in ok], "invalid_citations": bad}

    def respond(state: State):
        return {"history": [{"role": "user", "content": state["question"]},
                            {"role": "assistant", "content": state["answer"]}]}

    g = StateGraph(State)
    for fn in (classify_intent, retrieve_docs, call_mcp_tools, validate_evidence, synthesize,
               verify_citations, respond):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "classify_intent")
    g.add_conditional_edges("classify_intent", after_classify,
                            ["retrieve_docs", "call_mcp_tools", "validate_evidence"])
    g.add_conditional_edges("retrieve_docs", after_docs, ["call_mcp_tools", "validate_evidence"])
    g.add_edge("call_mcp_tools", "validate_evidence")
    g.add_edge("validate_evidence", "synthesize")
    g.add_edge("synthesize", "verify_citations")
    g.add_edge("verify_citations", "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)


RESULT_KEYS = ("answer", "citations", "invalid_citations", "live_sources", "live", "routes")


class Agent:
    """memory=True runs on the checkpointed graph (history by thread_id); memory=False on a
    checkpointer-free graph, so nothing is read or stored."""

    def __init__(self, with_memory, without_memory):
        self._mem, self._plain = with_memory, without_memory

    def _run(self, question: str, thread_id: str, memory: bool):
        graph = self._mem if (memory and self._mem is not None) else self._plain
        cfg = {"configurable": {"thread_id": thread_id}} if graph is self._mem else {}
        return graph, {"question": question}, cfg

    async def ainvoke(self, question: str, thread_id: str, memory: bool = True) -> dict:
        graph, inp, cfg = self._run(question, thread_id, memory)
        state = await graph.ainvoke(inp, cfg)
        return {k: state.get(k) for k in RESULT_KEYS}

    def astream(self, question: str, thread_id: str, memory: bool = True):
        """Yields (mode, payload): 'updates' = per-node state updates, 'custom' = token events."""
        graph, inp, cfg = self._run(question, thread_id, memory)
        return graph.astream(inp, cfg, stream_mode=["updates", "custom"])


def build_agent(router=None, tools=None, checkpointer=None) -> Agent:
    router = router or router_mod.load_router()
    return Agent(
        build_graph(router, tools, checkpointer) if checkpointer is not None else None,
        build_graph(router, tools, None),
    )
