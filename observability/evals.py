"""16-variant RAG experiment: 4 chunkers x 2 retrieval modes x 2 rerank modes.

Every variant answers the same golden questions through the same pipeline; only the chunker's
ingest run, the retrieval mode and the rerank switch differ. Scores:
  deterministic: Recall@k / MRR (expected source ids), citation correctness, keyword coverage,
                 correct refusal (unanswerable questions)
  LLM-judged (RAGAS-style, implemented in-house so every LLM call goes through src.llm):
                 faithfulness, answer relevancy, context precision, context recall
Selection: top candidates on the dev split are confirmed on the held-out split.
"""
import hashlib
import itertools
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from src import config, llm, prompts, retrieval

CHUNKERS = ("hierarchical", "fixed50", "recursive200", "hybrid128")
MODES = ("dense", "hybrid")
REFUSAL_PHRASES = ("cannot answer", "can't answer", "not in the provided", "outside what i can")


@dataclass(frozen=True)
class Variant:
    chunker: str
    mode: str
    rerank: bool

    @property
    def name(self) -> str:
        return f"{self.chunker}|{self.mode}|{'rerank' if self.rerank else 'norerank'}"


def all_variants() -> list[Variant]:
    return [Variant(c, m, r) for c, m, r in itertools.product(CHUNKERS, MODES, (False, True))]


# ---- the pipeline under test --------------------------------------------------------------------

def answer_with_variant(question: str, variant: Variant, run_ids: dict[str, int]) -> dict:
    cfg = config.settings
    t0 = time.perf_counter()
    retrieved = retrieval.retrieve(question, run_id=run_ids[variant.chunker], mode=variant.mode,
                                   rerank=variant.rerank, top_n=cfg.evals.recall_k)
    contexts = [c for c in retrieved if c.get("content")][:cfg.retrieval.top_n]
    if not contexts:
        return {"answer": prompts.NO_CONTEXT_ANSWER, "retrieved": retrieved, "contexts": [], "citations": [],
                "invalid_citations": [], "latency_ms": int((time.perf_counter() - t0) * 1000), "cost": 0.0}
    out = llm.complete([
        {"role": "system", "content": prompts.SYSTEM_RAG},
        {"role": "user", "content": f"Context:\n{prompts.context_block(contexts)}\n\nQuestion: {question}"},
    ], temperature=0)
    ok, bad = prompts.extract_citations(out["content"] or "", len(contexts))
    return {
        "answer": out["content"] or "",
        "retrieved": retrieved,
        "contexts": contexts,
        "citations": [{**contexts[n - 1], "n": n} for n in ok],
        "invalid_citations": bad,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "cost": float((out.get("usage") or {}).get("cost") or 0.0),
    }


# ---- deterministic evaluators (outputs, expected) -> float --------------------------------------

def _expected(expected: dict) -> set[str]:
    return set(expected.get("source_ids", []))


def recall_at_k(outputs: dict, expected: dict, k: int) -> float:
    want = _expected(expected)
    got = {c["source_id"] for c in outputs["retrieved"][:k]}
    return len(want & got) / len(want)


def mrr(outputs: dict, expected: dict) -> float:
    want = _expected(expected)
    return next((1 / (i + 1) for i, c in enumerate(outputs["retrieved"]) if c["source_id"] in want), 0.0)


def citation_correct(outputs: dict, expected: dict) -> float:
    cites = outputs["citations"]
    return float(bool(cites) and not outputs["invalid_citations"] and any(c["source_id"] in _expected(expected) for c in cites))


def keyword_coverage(outputs: dict, expected: dict) -> float:
    kws = expected["keywords"]
    text = outputs["answer"].lower()
    return sum(k.lower() in text for k in kws) / len(kws)


def correct_refusal(outputs: dict, expected: dict) -> float:
    a = outputs["answer"].lower()
    return float(not outputs["citations"] and (outputs["answer"] == prompts.NO_CONTEXT_ANSWER or any(p in a for p in REFUSAL_PHRASES)))


# ---- LLM judges (all calls go through src.llm) ---------------------------------------------------

def _judge(system: str, user: str) -> dict | None:
    try:
        out = llm.complete([{"role": "system", "content": system + " Reply with JSON only."},
                            {"role": "user", "content": user}], temperature=0)
        m = re.search(r"\{.*\}", out["content"] or "", re.S)
        data = json.loads(m.group(0)) if m else None
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _ctx(outputs: dict) -> str:
    return "\n\n".join(f"[{i}] {c['content']}" for i, c in enumerate(outputs["contexts"], 1))


def _unit(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1 else None


def faithfulness(outputs: dict, example: dict) -> float | None:
    d = _judge('Count the factual claims in ANSWER ("total") and how many of them are supported by CONTEXT '
               '("supported"). Output {"supported": n, "total": m}.',
               f"CONTEXT:\n{_ctx(outputs)}\n\nANSWER:\n{outputs['answer']}")
    try:
        s, t = d["supported"], d["total"]
        return s / t if isinstance(s, int) and isinstance(t, int) and t > 0 and 0 <= s <= t else None
    except (TypeError, KeyError):
        return None


def answer_relevancy(outputs: dict, example: dict) -> float | None:
    d = _judge('Rate from 0 to 1 how directly ANSWER addresses QUESTION (1 = fully on point, 0 = irrelevant). '
               'Output {"score": x}.', f"QUESTION: {example['question']}\n\nANSWER:\n{outputs['answer']}")
    return _unit(d.get("score")) if d else None


def context_precision(outputs: dict, example: dict) -> float | None:
    d = _judge('For each numbered CONTEXT passage say whether it is relevant to answering QUESTION. '
               'Output {"relevant": [true/false per passage, in order]}.',
               f"QUESTION: {example['question']}\n\nCONTEXT:\n{_ctx(outputs)}")
    rel = d.get("relevant") if d else None
    if not isinstance(rel, list) or len(rel) != len(outputs["contexts"]) or not rel or not all(isinstance(x, bool) for x in rel):
        return None
    return sum(rel) / len(rel)


def context_recall(outputs: dict, example: dict) -> float | None:
    if not example.get("reference_answer"):
        return None
    d = _judge('Rate from 0 to 1 how much of REFERENCE is supported by the CONTEXT passages. Output {"score": x}.',
               f"REFERENCE:\n{example['reference_answer']}\n\nCONTEXT:\n{_ctx(outputs)}")
    return _unit(d.get("score")) if d else None


# ---- aggregation ---------------------------------------------------------------------------------

def summarize(values: list, seed: int = 0, n_boot: int | None = None) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    n_boot = n_boot or config.settings.evals.bootstrap_samples
    arr = np.asarray(vals, dtype=float)
    rng = np.random.RandomState(seed)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return {"n": len(vals), "mean": float(arr.mean()),
            "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5))}


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))


DOC_METRICS = ("recall@5", "mrr", "citation_correct", "keyword_coverage",
               "faithfulness", "answer_relevancy", "context_precision", "context_recall")
COMPOSITE = ("recall@5", "citation_correct", "faithfulness")


def _score(item: dict, out: dict) -> dict:
    e, ex = item["expected"], {"question": item["question"], "reference_answer": item.get("reference_answer")}
    k = config.settings.evals.recall_k
    if item["category"] == "doc_qa":
        return {"recall@5": recall_at_k(out, e, k), "mrr": mrr(out, e), "citation_correct": citation_correct(out, e),
                "keyword_coverage": keyword_coverage(out, e), "faithfulness": faithfulness(out, ex),
                "answer_relevancy": answer_relevancy(out, ex), "context_precision": context_precision(out, ex),
                "context_recall": context_recall(out, ex)}
    return {"correct_refusal": correct_refusal(out, e)}


def _composite(summary: dict) -> float | None:
    means = [summary[m]["mean"] for m in COMPOSITE if summary.get(m, {}).get("mean") is not None]
    return sum(means) / len(means) if means else None


def _retrieval_latencies(variant: Variant, items: list[dict], run_ids: dict[str, int]) -> list[float]:
    """Sequential (no worker contention) wall time of retrieval + rerank only, ms per question. That is the
    part that differs between variants; the LLM call costs the same for all of them."""
    out = []
    for it in items:
        t0 = time.perf_counter()
        retrieval.retrieve(it["question"], run_id=run_ids[variant.chunker], mode=variant.mode,
                           rerank=variant.rerank, top_n=config.settings.evals.recall_k)
        out.append((time.perf_counter() - t0) * 1000)
    return out


def run_experiment(items: list[dict], run_ids: dict[str, int], *, workers: int | None = None,
                   variants: list[Variant] | None = None, timer=None) -> dict:
    """Score every variant on every doc_qa/unanswerable item. -> {experiment_id, variants, winner}."""
    workers = workers or config.settings.evals.workers
    variants = variants or all_variants()
    items = [i for i in items if i["category"] in ("doc_qa", "unanswerable")]

    def one(job):
        v, item = job
        out = answer_with_variant(item["question"], v, run_ids)
        return v, item, out, _score(item, out)

    jobs = [(v, i) for v in variants for i in items]
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(one, jobs))

    rows = []
    for v in variants:
        mine = [(i, o, s) for vv, i, o, s in results if vv == v]
        row = {"variant": v.name, "chunker": v.chunker, "mode": v.mode, "rerank": v.rerank, "run_id": run_ids[v.chunker]}
        for split in ("dev", "heldout"):
            part = [(i, o, s) for i, o, s in mine if i["split"] == split]
            summ = {m: summarize([s.get(m) for i, o, s in part if i["category"] == "doc_qa"]) for m in DOC_METRICS}
            summ["correct_refusal"] = summarize([s["correct_refusal"] for i, o, s in part if i["category"] == "unanswerable"])
            summ["composite"] = _composite(summ)
            row[split] = summ
        lat = [o["latency_ms"] for _, o, _ in mine]
        rlat = (timer or _retrieval_latencies)(v, [i for i in items if i["category"] == "doc_qa" and i["split"] == "dev"], run_ids)
        row.update(latency_p50_ms=percentile(lat, 50), latency_p95_ms=percentile(lat, 95),
                   retrieval_p50_ms=percentile(rlat, 50), retrieval_p95_ms=percentile(rlat, 95),
                   cost_per_query=sum(o["cost"] for _, o, _ in mine) / len(mine))
        rows.append(row)

    dev = [{"variant": r["variant"], "composite": r["dev"]["composite"] or 0.0, "cost_per_query": r["cost_per_query"],
            "latency_p95_ms": r["retrieval_p95_ms"]} for r in rows]
    held = {r["variant"]: {"variant": r["variant"], "composite": r["heldout"]["composite"] or 0.0,
                           "cost_per_query": r["cost_per_query"], "latency_p95_ms": r["retrieval_p95_ms"]} for r in rows}
    key = json.dumps({"items": [i["id"] for i in items], "runs": run_ids, "variants": [v.name for v in variants],
                      "model": config.settings.llm["openrouter"]["model"], "top_n": config.settings.retrieval.top_n,
                      "k": config.settings.retrieval.k}, sort_keys=True)
    return {"experiment_id": "exp-" + hashlib.sha256(key.encode()).hexdigest()[:10],
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_items": {s: sum(1 for i in items if i["split"] == s) for s in ("dev", "heldout")},
            "variants": rows, "winner": select_winner(dev, held, top=3)}


def select_winner(dev_rows: list[dict], heldout: dict[str, dict], top: int = 3) -> dict:
    """Top `top` variants by dev composite are compared on the held-out split; ties go to the cheaper,
    then the faster variant."""
    cands = sorted(dev_rows, key=lambda r: -r["composite"])[:top]
    best = min((heldout[c["variant"]] for c in cands),
               key=lambda h: (-h["composite"], h["cost_per_query"], h["latency_p95_ms"]))
    dev = next(r for r in dev_rows if r["variant"] == best["variant"])
    return {"variant": best["variant"], "heldout_composite": best["composite"], "dev_composite": dev["composite"],
            "cost_per_query": best["cost_per_query"], "latency_p95_ms": best["latency_p95_ms"]}


# ---- run/ingest bookkeeping + config update --------------------------------------------------------

def resolve_runs(conn) -> dict[str, int]:
    """chunker -> run_id for the corpus revision of the active run (all four must exist and be valid)."""
    row = conn.execute("SELECT corpus_revision FROM ingest_runs WHERE status='active'").fetchone()
    if not row:
        raise RuntimeError("no active ingest run; run standalone-scripts/ingest.py first")
    rows = dict(conn.execute(
        "SELECT chunk_strategy, run_id FROM ingest_runs WHERE corpus_revision=%s AND status IN ('active','validated')", (row[0],)).fetchall())
    missing = [c for c in CHUNKERS if c not in rows]
    if missing:
        raise RuntimeError(f"corpus {row[0]} has no valid run for chunker(s) {missing}; ingest them with --strategy")
    return {c: rows[c] for c in CHUNKERS}


def apply_winner(v: Variant, experiment_id: str, path=None) -> None:
    """Point config.yaml's retrieval selection at the winner. Only chunker/mode/rerank change; the
    reranker model name stays, and an `eval-winner` marker records which experiment chose it."""
    path = path or config.ROOT / "config" / "config.yaml"
    text = path.read_text()
    head, rest = text.split("\nretrieval:\n", 1)
    block, tail = re.split(r"(?=\n\S)", rest, maxsplit=1)
    block = re.sub(r"^  # eval-winner:.*\n", "", block, flags=re.M)
    for key, val in (("mode", f'"{v.mode}"'), ("chunker", f'"{v.chunker}"'), ("rerank", "true" if v.rerank else "false")):
        block, n = re.subn(rf"^(  {key}:\s*)\S+", lambda m, val=val: m.group(1) + val, block, count=1, flags=re.M)
        assert n == 1, f"retrieval.{key} not found in config.yaml"
    block = f"  # eval-winner: {experiment_id} ({v.name})\n" + block.lstrip("\n")
    path.write_text(head + "\nretrieval:\n" + block + tail)


def leaderboard_markdown(res: dict) -> str:
    def f(x):
        return "—" if x is None else f"{x:.3f}"

    def cell(s, m):
        d = s[m]
        return "—" if d["mean"] is None else f"{d['mean']:.2f} [{d['ci_low']:.2f}, {d['ci_high']:.2f}]"

    lines = [f"# RAG variant leaderboard `{res['experiment_id']}`", "",
             f"Items: {res['n_items']}. Winner: **{res['winner']['variant']}** "
             f"(held-out composite {f(res['winner']['heldout_composite'])}).", "",
             "Latency: retrieval = sequential retrieval+rerank only; end-to-end = whole answer under 4 parallel workers (noisy).", "",
             "| variant | dev composite | held-out composite | Recall@5 (dev) | citation (dev) | faithfulness (dev) | retrieval p50/p95 ms | e2e p50/p95 ms | $/query |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(res["variants"], key=lambda r: -(r["dev"]["composite"] or 0)):
        lines.append(f"| {r['variant']} | {f(r['dev']['composite'])} | {f(r['heldout']['composite'])} | "
                     f"{cell(r['dev'], 'recall@5')} | {cell(r['dev'], 'citation_correct')} | {cell(r['dev'], 'faithfulness')} | "
                     f"{r['retrieval_p50_ms']:.0f} / {r['retrieval_p95_ms']:.0f} | {r['latency_p50_ms']:.0f} / {r['latency_p95_ms']:.0f} | {r['cost_per_query']:.5f} |")
    return "\n".join(lines) + "\n"
