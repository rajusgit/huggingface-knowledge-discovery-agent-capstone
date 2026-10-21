"""LangGraph agent (hand-built nodes; every LLM call goes through src.llm).

START -> guard_input -> classify_intent -+-> [retrieve_docs] -> [call_mcp_tools] -+-> validate_evidence
                          |                                        |
                          +-> plan_benchmark -> authorize_benchmark (human approval, interrupt)
                                -> enqueue_job -> poll_results ----+
      validate_evidence -> synthesize -> verify_citations -> respond -> END

Tool outputs and retrieved text are untrusted data: delimited, never executed as instructions.
Benchmarks never run without explicit human approval. Memory = `history` in state, persisted by
the checkpointer keyed by thread_id; memory-off turns ignore history and delete their state.
"""
import asyncio
import json
import operator
import re
import time
from typing import Annotated, TypedDict

from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents import guards as guards_mod
from agents import router as router_mod
from src import config, llm, prompts, retrieval
from tools import benchmarks

TERMINAL = {"succeeded", "failed", "timeout"}


class State(TypedDict, total=False):
    question: str
    use_history: bool
    history: Annotated[list[dict], operator.add]
    routes: list[str]
    chunks: list[dict]
    live: list[dict]
    bench_plan: dict
    bench_jobs: list[str]
    benchmarks: list[dict]
    refused: bool
    blocked: bool
    guard_events: list[dict]
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


def _json_object(text: str | None) -> dict:
    """Parse a JSON object out of an LLM reply (tolerates code fences / chatter)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _refuse(message: str) -> dict:
    return {"refused": True, "answer": message}


def build_graph(router, tools, checkpointer, guard):
    cfg = config.settings.agent
    bcfg = config.settings.benchmarks
    tools = tools or []
    by_name = {t.name: t for t in tools}
    openai_tools = [convert_to_openai_tool(t) for t in tools]

    def hist(state: State) -> list[dict]:
        return state.get("history", [])[-2 * cfg.history_turns:] if state.get("use_history", True) else []

    def guard_input(state: State):
        res = guard.check(state["question"])
        update = {"question": res.text, "guard_events": res.events, "blocked": res.blocked}
        if res.blocked:  # skip the whole pipeline; the blocked text is never kept as context
            update |= {"routes": [], "chunks": [], "live": [], "benchmarks": [], "citations": [], "invalid_citations": [],
                       "live_sources": [], "bench_plan": {}, "bench_jobs": [], "refused": True, "answer": prompts.BLOCKED_ANSWER}
        return update

    def after_guard(state: State) -> str:
        return "validate_evidence" if state["blocked"] else "classify_intent"

    def classify_intent(state: State):
        intent = router.classify(state["question"])
        return {"routes": intent.routes, "chunks": [], "live": [], "refused": False, "answer": "",
                "citations": [], "invalid_citations": [], "live_sources": [], "bench_plan": {},
                "bench_jobs": [], "benchmarks": []}

    def after_classify(state: State) -> str:
        if "evaluate" in state["routes"]:
            return "plan_benchmark"
        docs, mcp = router_mod.branches(state["routes"])
        return "retrieve_docs" if docs else "call_mcp_tools" if mcp else "validate_evidence"

    def after_docs(state: State) -> str:
        return "call_mcp_tools" if router_mod.branches(state["routes"])[1] else "validate_evidence"

    def unless_refused(nxt: str):
        return lambda state: "validate_evidence" if state.get("refused") else nxt

    def retrieve_docs(state: State):
        hits = retrieval.retrieve(state["question"])
        return {"chunks": [c for c in hits if c.get("content")]}

    async def call_mcp_tools(state: State):
        if not by_name:
            return {"live": []}
        messages = [{"role": "system", "content": prompts.SYSTEM_TOOLS}, *hist(state),
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

    # ---- benchmarks: plan -> human approval -> enqueue -> poll -------------------------------

    def plan_benchmark(state: State):
        datasets = benchmarks.list_datasets()
        if not datasets:
            return _refuse("No benchmark datasets are available yet, so I can't run a benchmark.")
        system = prompts.SYSTEM_PLAN_BENCH.replace("{models}", ", ".join(bcfg.allowed_models)) \
                                          .replace("{datasets}", ", ".join(datasets))
        out = llm.complete([{"role": "system", "content": system}, *hist(state),
                            {"role": "user", "content": state["question"]}], temperature=0)
        data = _json_object(out["content"])
        # LLM output is untrusted: only allowlisted models / existing datasets survive.
        models = [m for m in dict.fromkeys(data.get("models") or [])
                  if isinstance(m, str) and m in bcfg.allowed_models][:bcfg.max_models_per_benchmark]
        dataset = data.get("dataset") if data.get("dataset") in datasets else (datasets[0] if len(datasets) == 1 else None)
        if not models or not dataset:
            return _refuse("Tell me which models and which dataset to benchmark. "
                           f"Allowed models: {', '.join(bcfg.allowed_models)}. Available datasets: {', '.join(datasets)}.")
        return {"bench_plan": {"models": models, "dataset": dataset, "seed": bcfg.default_seed}}

    def authorize_benchmark(state: State):
        plan = state["bench_plan"]
        decision = interrupt({
            "type": "benchmark_approval", **plan,
            "limits": {"max_runtime_seconds": bcfg.max_runtime_seconds, "max_memory_mb": bcfg.max_memory_mb},
        })
        if not (isinstance(decision, dict) and decision.get("approved") is True):
            return _refuse("The benchmark was not approved, so nothing was run.")
        return {}

    def enqueue_job(state: State):
        plan = state["bench_plan"]
        jobs = []
        try:
            for model in plan["models"]:
                jid = benchmarks.create_job(model, plan["dataset"], plan["seed"])
                jobs.append(str(jid))
                benchmarks.start_in_background(jid)
        except benchmarks.BenchmarkRefused as e:
            return _refuse(f"The benchmark could not be started: {e}")
        return {"bench_jobs": jobs}

    async def poll_results(state: State):
        writer = get_stream_writer()
        jobs = state["bench_jobs"]
        deadline = time.monotonic() + bcfg.max_runtime_seconds * len(jobs) + 60
        while True:
            recs = [benchmarks.jsonable(await asyncio.to_thread(benchmarks.get_job, j)) for j in jobs]
            writer({"type": "benchmark_status",
                    "jobs": [{"job_id": r["job_id"], "model_id": r["model_id"], "status": r["status"]} for r in recs]})
            if all(r["status"] in TERMINAL for r in recs) or time.monotonic() > deadline:
                return {"benchmarks": recs}
            await asyncio.sleep(bcfg.poll_interval_seconds)

    # ---- answer ------------------------------------------------------------------------------

    def validate_evidence(state: State):
        if state.get("refused"):
            return {}
        if not state["routes"]:
            return _refuse(prompts.OUT_OF_SCOPE_ANSWER)
        if not state["chunks"] and not state["live"] and not state["benchmarks"]:
            return _refuse(prompts.NO_CONTEXT_ANSWER)
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
        if state["benchmarks"]:
            data = json.dumps(state["benchmarks"], default=str)[:cfg.max_tool_result_chars * 2]
            parts.append("MEASURED benchmark results (run in our sandbox; report status, errors and "
                         "provenance honestly):\n" + prompts.wrap_untrusted(data))
        messages = [{"role": "system", "content": prompts.SYSTEM_AGENT}, *hist(state),
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
        if state.get("blocked"):
            return {}
        return {"history": [{"role": "user", "content": state["question"]},
                            {"role": "assistant", "content": state["answer"]}]}

    g = StateGraph(State)
    for fn in (guard_input, classify_intent, retrieve_docs, call_mcp_tools, plan_benchmark, authorize_benchmark,
               enqueue_job, poll_results, validate_evidence, synthesize, verify_citations, respond):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", after_guard, ["classify_intent", "validate_evidence"])
    g.add_conditional_edges("classify_intent", after_classify,
                            ["retrieve_docs", "call_mcp_tools", "plan_benchmark", "validate_evidence"])
    g.add_conditional_edges("retrieve_docs", after_docs, ["call_mcp_tools", "validate_evidence"])
    g.add_edge("call_mcp_tools", "validate_evidence")
    g.add_conditional_edges("plan_benchmark", unless_refused("authorize_benchmark"),
                            ["authorize_benchmark", "validate_evidence"])
    g.add_conditional_edges("authorize_benchmark", unless_refused("enqueue_job"),
                            ["enqueue_job", "validate_evidence"])
    g.add_conditional_edges("enqueue_job", unless_refused("poll_results"), ["poll_results", "validate_evidence"])
    g.add_edge("poll_results", "validate_evidence")
    g.add_edge("validate_evidence", "synthesize")
    g.add_edge("synthesize", "verify_citations")
    g.add_edge("verify_citations", "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)


RESULT_KEYS = ("answer", "citations", "invalid_citations", "live_sources", "live", "routes", "benchmarks", "guard_events")


class Agent:
    """One checkpointed graph (needed so a benchmark can pause for approval).

    memory=True : history persists under thread_id.
    memory=False: history is ignored and the turn runs on a private derived thread whose state is
                  deleted once the turn finishes (a pending approval keeps it until resolved).
    """

    def __init__(self, graph, saver):
        self._graph, self._saver = graph, saver

    @staticmethod
    def _tid(thread_id: str, memory: bool) -> str:
        return thread_id if memory else f"{thread_id}~nomem"

    def _cfg(self, thread_id: str, memory: bool) -> dict:
        return {"configurable": {"thread_id": self._tid(thread_id, memory)}}

    async def _cleanup(self, thread_id: str, memory: bool, interrupted: bool) -> None:
        if not memory and not interrupted:
            await self._saver.adelete_thread(self._tid(thread_id, memory))

    async def _invoke(self, inp, thread_id: str, memory: bool) -> dict:
        state = await self._graph.ainvoke(inp, self._cfg(thread_id, memory))
        pending = state.get("__interrupt__")
        await self._cleanup(thread_id, memory, bool(pending))
        out = {k: state.get(k) for k in RESULT_KEYS}
        out["interrupt"] = pending[0].value if pending else None
        return out

    async def _stream(self, inp, thread_id: str, memory: bool):
        interrupted = False
        async for mode, payload in self._graph.astream(inp, self._cfg(thread_id, memory),
                                                       stream_mode=["updates", "custom"]):
            if mode == "updates" and "__interrupt__" in payload:
                interrupted = True
                yield "interrupt", payload["__interrupt__"][0].value
            else:
                yield mode, payload
        await self._cleanup(thread_id, memory, interrupted)

    async def ainvoke(self, question: str, thread_id: str, memory: bool = True) -> dict:
        return await self._invoke({"question": question, "use_history": memory}, thread_id, memory)

    def astream(self, question: str, thread_id: str, memory: bool = True):
        """Yields (mode, payload): 'updates' per-node state updates, 'custom' token/status events,
        'interrupt' when human approval is required."""
        return self._stream({"question": question, "use_history": memory}, thread_id, memory)

    async def aresume(self, thread_id: str, approved: bool, memory: bool = True) -> dict:
        return await self._invoke(Command(resume={"approved": approved}), thread_id, memory)

    def aresume_stream(self, thread_id: str, approved: bool, memory: bool = True):
        return self._stream(Command(resume={"approved": approved}), thread_id, memory)


def build_agent(router=None, tools=None, checkpointer=None, guard=None) -> Agent:
    saver = checkpointer if checkpointer is not None else InMemorySaver()
    return Agent(build_graph(router or router_mod.load_router(), tools, saver, guard or guards_mod.load_guard()), saver)
