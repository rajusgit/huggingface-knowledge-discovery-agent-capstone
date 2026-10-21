"""Sandboxed embedding-model retrieval benchmark.

Parent (this module, imported by the MCP server / API): validates, records the job in
hf_discovery.benchmark_runs, runs the worker in a SUBPROCESS and stores the outcome.
Worker (`python -m tools.benchmarks --worker ...`): loads the model on CPU, never with
trust_remote_code, scores a retrieval dataset. Allowlisted models/datasets only.

Dataset file (data/benchmarks/<id>.json):
  {"id", "revision", "corpus": {doc_id: text}, "queries": {qid: text}, "qrels": {qid: [doc_id]}}
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from src import config
from tools import hf_client

MAX_ERROR_CHARS = 2000
_ENV_KEEP = ("PATH", "HOME", "HF_HOME", "TMPDIR", "LANG", "LC_ALL", "VIRTUAL_ENV", "SYSTEMROOT")


class BenchmarkRefused(ValueError):
    pass


# ---- dataset + metrics (pure) ----------------------------------------------------

def dataset_path(dataset_id: str) -> Path:
    if not isinstance(dataset_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", dataset_id):
        raise BenchmarkRefused(f"invalid dataset id: {dataset_id!r}")
    path = config.ROOT / config.settings.benchmarks.dataset_dir / f"{dataset_id}.json"
    if not path.is_file():
        raise BenchmarkRefused(f"unknown benchmark dataset: {dataset_id!r}")
    return path


def list_datasets() -> list[str]:
    d = config.ROOT / config.settings.benchmarks.dataset_dir
    return sorted(p.stem for p in d.glob("*.json") if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", p.stem)) if d.is_dir() else []


def jsonable(rec: dict) -> dict:
    """DB record -> plain JSON types (uuid/datetime to str)."""
    return {k: (str(v) if isinstance(v, uuid.UUID) else v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in rec.items()}


def start_in_background(job: uuid.UUID) -> None:
    # ponytail: in-process daemon thread; a job queue (RQ/Celery) if jobs must survive restarts
    threading.Thread(target=run_job, args=(job,), daemon=True, name=f"benchmark-{job}").start()


def load_dataset(path: Path) -> dict:
    ds = json.loads(Path(path).read_text())
    missing = {"id", "revision", "corpus", "queries", "qrels"} - set(ds)
    if missing:
        raise BenchmarkRefused(f"dataset missing keys {sorted(missing)}")
    cfg = config.settings.benchmarks
    if len(ds["corpus"]) > cfg.max_corpus_docs or len(ds["queries"]) > cfg.max_queries:
        raise BenchmarkRefused("dataset exceeds configured size limits")
    ds["qrels"] = {q: r for q, r in ds["qrels"].items() if q in ds["queries"] and r}
    ds["queries"] = {q: t for q, t in ds["queries"].items() if q in ds["qrels"]}
    if not ds["queries"]:
        raise BenchmarkRefused("dataset has no queries with relevance judgements")
    return ds


def compute_metrics(ranked: dict[str, list[str]], qrels: dict[str, list[str]], ks=(1, 5, 10)) -> dict:
    """Binary-relevance Recall@k, MRR@10, nDCG@10, averaged over queries."""
    out = {f"recall@{k}": 0.0 for k in ks} | {"mrr@10": 0.0, "ndcg@10": 0.0}
    for qid, rel in qrels.items():
        rel, order = set(rel), ranked[qid]
        for k in ks:
            out[f"recall@{k}"] += len(rel & set(order[:k])) / len(rel)
        rr = next((1 / (i + 1) for i, d in enumerate(order[:10]) if d in rel), 0.0)
        dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(order[:10]) if d in rel)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))
        out["mrr@10"] += rr
        out["ndcg@10"] += dcg / ideal
    return {k: v / len(qrels) for k, v in out.items()}


def evaluate(encode_queries, encode_docs, ds: dict, ks=(1, 5, 10), seed: int = 0) -> dict:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    doc_ids, doc_texts = list(ds["corpus"]), list(ds["corpus"].values())
    q_ids, q_texts = list(ds["queries"]), list(ds["queries"].values())

    def unit(a):
        a = np.asarray(a, dtype=float)
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    t0 = time.perf_counter()
    docs = unit(encode_docs(doc_texts))
    doc_seconds = time.perf_counter() - t0
    queries = unit(encode_queries(q_texts))
    t1 = time.perf_counter()
    sample = q_texts[:50]
    for t in sample:  # single-query latency
        encode_queries([t])
    q_latency_ms = (time.perf_counter() - t1) / len(sample) * 1000
    sims = queries @ docs.T
    ranked = {q: [doc_ids[j] for j in np.argsort(-sims[i], kind="stable")[:max(max(ks), 10)]]
              for i, q in enumerate(q_ids)}
    return {
        "metrics": compute_metrics(ranked, ds["qrels"], ks),
        "timing": {"corpus_encode_seconds": round(doc_seconds, 4),
                   "docs_per_second": round(len(doc_ids) / doc_seconds, 2) if doc_seconds else None,
                   "query_latency_ms": round(q_latency_ms, 3)},
        "n_docs": len(doc_ids),
        "n_queries": len(q_ids),
    }


# ---- worker (runs in the subprocess) --------------------------------------------

def _worker(args) -> None:
    import platform
    import resource

    import numpy as np
    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(args.seed)
    prefixes = json.loads(args.prefixes)
    model = SentenceTransformer(args.model, revision=args.revision, device="cpu", trust_remote_code=False)
    ds = json.loads(Path(args.dataset).read_text())  # parent already validated it

    def enc(prefix):
        return lambda texts: model.encode([prefix + t for t in texts], convert_to_numpy=True,
                                          show_progress_bar=False)

    res = evaluate(enc(prefixes.get("query", "")), enc(prefixes.get("passage", "")), ds,
                   ks=json.loads(args.ks), seed=args.seed)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    res["peak_rss_mb"] = round(rss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    res["hardware"] = {
        "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "python": platform.python_version(), "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__, "numpy": np.__version__, "device": "cpu",
    }
    print(json.dumps(res))


# ---- parent: job lifecycle --------------------------------------------------------

def _worker_cmd(model_id: str, revision: str | None, dataset_file: Path, seed: int) -> list[str]:
    cfg = config.settings.benchmarks
    return [sys.executable, "-m", "tools.benchmarks", "--worker", "--model", model_id,
            "--revision", revision or "", "--dataset", str(dataset_file), "--seed", str(seed),
            "--ks", json.dumps(cfg.ks), "--prefixes", json.dumps(cfg.prefixes.get(model_id, {}))]


def _limits():
    if sys.platform.startswith("linux"):  # RLIMIT_AS is unreliable on macOS; checked post-run there
        import resource
        cap = config.settings.benchmarks.max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def _conn():
    import psycopg

    return psycopg.connect(config.settings.dsn, options=f"-c search_path={config.settings.database.schema_name},public")


def create_job(model_id: str, dataset_id: str, seed: int | None = None) -> uuid.UUID:
    cfg = config.settings.benchmarks
    if model_id not in cfg.allowed_models:
        raise BenchmarkRefused(f"model {model_id!r} is not on the benchmark allowlist")
    ds = load_dataset(dataset_path(dataset_id))
    try:
        revision = hf_client.get_model_details(model_id)["revision"]
    except hf_client.HFError as e:
        raise BenchmarkRefused(f"could not resolve a revision for {model_id}: {e}") from e
    job = uuid.uuid4()
    seed = cfg.default_seed if seed is None else seed
    with _conn() as c:
        c.execute(
            """INSERT INTO benchmark_runs (job_id, model_id, model_revision, dataset_id, dataset_revision, status, seed)
               VALUES (%s,%s,%s,%s,%s,'queued',%s)""",
            (job, model_id, revision, dataset_id, ds["revision"], seed))
    return job


def _finish(job: uuid.UUID, status: str, metrics=None, hardware=None, error=None) -> None:
    from psycopg.types.json import Jsonb

    with _conn() as c:
        c.execute("""UPDATE benchmark_runs SET status=%s, metrics=%s, hardware=%s, error=%s, finished_at=now()
                     WHERE job_id=%s""",
                  (status, Jsonb(metrics) if metrics is not None else None,
                   Jsonb(hardware) if hardware is not None else None, error, job))


def run_job(job: uuid.UUID) -> None:
    """Run the worker for a queued job. Never raises: every outcome is recorded on the row."""
    cfg = config.settings.benchmarks
    with _conn() as c:
        c.execute("UPDATE benchmark_runs SET status='running' WHERE job_id=%s", (job,))
        model, revision, dataset_id, seed = c.execute(
            "SELECT model_id, model_revision, dataset_id, seed FROM benchmark_runs WHERE job_id=%s", (job,)).fetchone()
    try:
        env = {k: v for k, v in os.environ.items() if k in _ENV_KEEP}  # no API keys/secrets in the sandbox
        proc = subprocess.run(
            _worker_cmd(model, revision, dataset_path(dataset_id), seed),
            capture_output=True, text=True, timeout=cfg.max_runtime_seconds, env=env,
            cwd=config.ROOT, preexec_fn=_limits if sys.platform.startswith("linux") else None)
    except subprocess.TimeoutExpired:
        return _finish(job, "timeout", error=f"exceeded the {cfg.max_runtime_seconds}s runtime limit")
    except Exception as e:
        return _finish(job, "failed", error=f"could not start worker: {type(e).__name__}: {e}"[:MAX_ERROR_CHARS])
    if proc.returncode != 0:
        return _finish(job, "failed", error=(proc.stderr or f"exit code {proc.returncode}")[-MAX_ERROR_CHARS:])
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
    try:
        res = json.loads(line)
        metrics = {**res["metrics"], "timing": res["timing"], "n_docs": res["n_docs"],
                   "n_queries": res["n_queries"], "peak_rss_mb": res.get("peak_rss_mb")}
    except Exception:
        return _finish(job, "failed", error="worker produced no valid result")
    if (res.get("peak_rss_mb") or 0) > cfg.max_memory_mb:
        return _finish(job, "failed", error=f"peak memory {res['peak_rss_mb']} MB exceeded {cfg.max_memory_mb} MB")
    _finish(job, "succeeded", metrics=metrics, hardware=res.get("hardware"))


def get_job(job: uuid.UUID | str) -> dict | None:
    from psycopg.rows import dict_row

    with _conn() as c:
        c.row_factory = dict_row
        return c.execute("SELECT * FROM benchmark_runs WHERE job_id=%s", (job,)).fetchone()


def benchmark_embedding_model(model_id: str, dataset_id: str, seed: int | None = None) -> dict:
    """Validate, run synchronously, return the stored record (status says what happened)."""
    job = create_job(model_id, dataset_id, seed)
    run_job(job)
    return get_job(job)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="internal worker entry point")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--model")
    p.add_argument("--revision", default="")
    p.add_argument("--dataset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ks", default="[1,5,10]")
    p.add_argument("--prefixes", default="{}")
    a = p.parse_args()
    if a.worker:
        a.revision = a.revision or None
        _worker(a)
    else:
        sys.exit("usage: python -m tools.benchmarks --worker ... (jobs are started via benchmark_embedding_model)")
