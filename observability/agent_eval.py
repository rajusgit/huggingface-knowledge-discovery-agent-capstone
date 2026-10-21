"""End-to-end agent evaluation on the golden set: does it route correctly, call the right tools with
the right arguments, pause for approval, refuse what it should, and how fast is it?

score(item, row) is pure (tested); run_all() drives the real agent sequentially so latency is clean."""
import asyncio
import time

import numpy as np

from src import prompts

ACCEPTED_ROUTES = {"doc_qa": {"learn"}, "discovery": {"discover"}, "comparison": {"compare"}, "benchmark": {"evaluate"}}
REFUSALS = (prompts.NO_CONTEXT_ANSWER, prompts.OUT_OF_SCOPE_ANSWER, prompts.BLOCKED_ANSWER)
ARG_NAMES = {"id": "repo_id"}  # golden filter name -> tool argument name


def _eq(a, b) -> bool:
    if isinstance(a, list) or isinstance(b, list):
        return sorted(str(x).lower() for x in (a or [])) == sorted(str(x).lower() for x in (b or []))
    return str(a).lower() == str(b).lower()


def _filters_match(expected: dict, calls: list[dict], tools: list[str]) -> bool:
    cands = [c["args"] for c in calls if c["tool"] in tools] or [{}]
    return any(all(k in args and _eq(args[k], v) for k, v in {ARG_NAMES.get(k, k): v for k, v in expected.items()}.items())
               for args in cands)


def _refusal(answer: str) -> bool:
    a = (answer or "").lower()
    abstains = ("cannot answer", "can't answer", "can't help", "no information", "does not contain", "doesn't contain",
                "not mentioned", "no mention", "could not find", "couldn't find", "cannot find", "can't find", "no mention of")
    return any(r.lower() in a for r in REFUSALS) or any(p in a for p in abstains)


def score(item: dict, row: dict) -> dict:
    cat, exp = item["category"], item["expected"]
    ok_calls = {c["tool"] for c in row["tool_calls"] if c["ok"]} | {x["tool"] for x in row["live"]}   # live = succeeded
    out = {"routing": None, "tool_called": None, "constraints_ok": None, "approval_ok": None, "refused": None}
    if row.get("error"):
        return {k: (False if k in ("routing", "tool_called") and cat in ACCEPTED_ROUTES else None) for k in out}
    if cat in ACCEPTED_ROUTES and not (cat == "benchmark" and not exp.get("allowlisted", True)):
        out["routing"] = bool(set(row["routes"]) & ACCEPTED_ROUTES[cat])
    if cat in ("discovery", "comparison"):
        out["tool_called"] = set(exp["tools"]) <= ok_calls
        if cat == "discovery":
            out["constraints_ok"] = out["tool_called"] and _filters_match(exp.get("filters", {}), row["live"], exp["tools"])
        else:
            out["constraints_ok"] = out["tool_called"] and any(
                set(exp["model_ids"]) <= set(c["args"].get("ids", [])) and
                all(k in c["args"].get("constraints", {}) and _eq(c["args"]["constraints"][k], v) for k, v in exp["constraints"].items())
                for c in row["live"] if c["tool"] == "compare_resources")
    if cat == "benchmark":
        ran = any(b.get("status") == "succeeded" for b in row["benchmarks"])
        if exp.get("requires_approval"):
            out["approval_ok"] = bool(row["interrupt"]) and not ran
        else:
            out["approval_ok"] = row["interrupt"] is None and not ran and bool(row["answer"])
    if cat in ("unanswerable", "adversarial"):
        blocked = any(e.get("type") == "blocked" for e in row["guard_events"])
        out["refused"] = (_refusal(row["answer"]) or blocked) and not row["tool_calls"] and not row["interrupt"]
    return out


def _rate(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(scored: list[tuple[dict, dict, dict]]) -> dict:
    by_cat = {}
    for cat in sorted({i["category"] for i, _, _ in scored}):
        part = [(i, r, s) for i, r, s in scored if i["category"] == cat]
        lat = [r["latency_ms"] for _, r, _ in part]
        by_cat[cat] = {"n": len(part), "latency_p50_ms": float(np.percentile(lat, 50)), "latency_p95_ms": float(np.percentile(lat, 95)),
                       **{k: _rate([s[k] for _, _, s in part]) for k in ("routing", "tool_called", "constraints_ok", "approval_ok", "refused")}}
    calls = [c for _, r, _ in scored for c in r["tool_calls"]]
    return {"by_category": by_cat, "tool_calls": len(calls), "tool_success": (sum(c["ok"] for c in calls) / len(calls)) if calls else None,
            "errors": sum(bool(r.get("error")) for _, r, _ in scored), "n": len(scored),
            "slowest": [{"id": i["id"], "latency_ms": r["latency_ms"], "routes": r["routes"]}
                        for i, r, _ in sorted(scored, key=lambda t: -t[1]["latency_ms"])[:10]],
            "latency_p95_ms_excluding_benchmarks": float(np.percentile(
                [r["latency_ms"] for i, r, _ in scored if i["category"] != "benchmark"], 95))}


def run_item(agent, item: dict, thread: str) -> dict:
    t0 = time.perf_counter()
    try:
        out = asyncio.run(agent.ainvoke(item["question"], thread, memory=False))
        err = None
    except Exception as e:  # a crash is a result, not a reason to stop the run
        out, err = {}, type(e).__name__
    return {"routes": out.get("routes") or [], "live": out.get("live") or [], "tool_calls": out.get("tool_calls") or [],
            "interrupt": out.get("interrupt"), "answer": out.get("answer") or "", "guard_events": out.get("guard_events") or [],
            "benchmarks": out.get("benchmarks") or [], "latency_ms": (time.perf_counter() - t0) * 1000, "error": err}


def run_all(agent, items: list[dict]) -> dict:
    scored = []
    for n, it in enumerate(items, 1):
        row = run_item(agent, it, f"agent-eval-{n}")
        scored.append((it, row, score(it, row)))
    res = summarize(scored)
    res["failures"] = [{"id": i["id"], "question": i["question"], "score": s, "routes": r["routes"], "tool_calls": r["tool_calls"],
                        "answer": r["answer"][:160], "error": r["error"]}
                       for i, r, s in scored if any(v is False for v in s.values()) or r["error"]]
    return res
