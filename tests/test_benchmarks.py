import json
import sys
import uuid

import psycopg
import pytest

from src import config
from tools import benchmarks as b

TINY = config.ROOT / "tests" / "fixtures" / "bench_tiny.json"
ALLOWED = "sentence-transformers/all-MiniLM-L6-v2"
_REAL_DETAILS = b.hf_client.get_model_details


# ---- pure metrics ------------------------------------------------------------

def test_metrics_on_a_hand_example():
    ranked = {"q1": ["a", "b", "c"], "q2": ["x", "y", "z"]}
    qrels = {"q1": ["b"], "q2": ["z", "w"]}
    m = b.compute_metrics(ranked, qrels, ks=(1, 3))
    assert m["recall@1"] == 0.0 and m["recall@3"] == pytest.approx((1.0 + 0.5) / 2)
    assert m["mrr@10"] == pytest.approx((1 / 2 + 1 / 3) / 2)
    import math
    assert m["ndcg@10"] == pytest.approx((1 / math.log2(3) + (1 / math.log2(4)) / (1 + 1 / math.log2(3))) / 2)


def fake_encode(texts):
    import hashlib
    import numpy as np
    out = []
    for t in texts:
        h = hashlib.sha256(t.lower().encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(float)[:16] - 127.5
        out.append(v)
    return np.array(out)


def test_evaluate_is_reproducible_for_a_seed():
    ds = b.load_dataset(TINY)
    r1 = b.evaluate(fake_encode, fake_encode, ds, ks=(1, 5), seed=3)
    r2 = b.evaluate(fake_encode, fake_encode, ds, ks=(1, 5), seed=3)
    assert r1["metrics"] == r2["metrics"] and r1["n_queries"] == 6 and r1["n_docs"] == 12
    assert r1["timing"]["docs_per_second"] > 0


# ---- guards ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_real_models_or_network(monkeypatch):
    monkeypatch.setattr(b.hf_client, "get_model_details", lambda i, revision=None: {"revision": "rev123"})


def test_model_outside_allowlist_is_refused_before_any_work(monkeypatch):
    monkeypatch.setattr(b.subprocess, "run", lambda *a, **k: pytest.fail("subprocess started"))
    with pytest.raises(b.BenchmarkRefused, match="allowlist"):
        b.create_job("evil/backdoored-model", "bench_tiny")


@pytest.mark.parametrize("ds", ["../secrets", "a/b", "", "bench tiny", "nope_not_there"])
def test_bad_or_missing_dataset_refused(ds):
    with pytest.raises(b.BenchmarkRefused):
        b.create_job(ALLOWED, ds)


# ---- job lifecycle (real DB; rows cleaned up) ---------------------------------

@pytest.fixture
def jobs(monkeypatch):
    monkeypatch.setattr(b, "dataset_path", lambda ds: TINY)  # use the fixture as 'bench_tiny'
    created = []
    orig = b.create_job

    def tracked(*a, **k):
        jid = orig(*a, **k)
        created.append(jid)
        return jid
    monkeypatch.setattr(b, "create_job", tracked)
    yield created
    if created:
        try:
            with psycopg.connect(config.settings.dsn, options="-c search_path=hf_discovery,public") as c:
                c.execute("DELETE FROM benchmark_runs WHERE job_id = ANY(%s)", (created,))
        except Exception:
            pass


def _db_ok():
    try:
        psycopg.connect(config.settings.dsn, connect_timeout=10).close()
        return True
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _db_ok(), reason="database unavailable")


def fake_worker(monkeypatch, code):
    monkeypatch.setattr(b, "_worker_cmd", lambda *a, **k: [sys.executable, "-c", code])


@needs_db
def test_success_path_stores_metrics_and_provenance(monkeypatch, jobs):
    payload = {"metrics": {"recall@5": 1.0}, "timing": {"docs_per_second": 5.0}, "hardware": {"cpu_count": 2},
               "n_docs": 12, "n_queries": 6, "peak_rss_mb": 100.0}
    fake_worker(monkeypatch, f"print('noise'); print({json.dumps(payload)!r})")
    rec = b.benchmark_embedding_model(ALLOWED, "bench_tiny", seed=7)
    assert rec["status"] == "succeeded" and rec["metrics"]["recall@5"] == 1.0
    assert rec["model_revision"] == "rev123" and rec["dataset_revision"] == "fixture-1" and rec["seed"] == 7
    assert rec["hardware"]["cpu_count"] == 2 and rec["finished_at"] is not None
    assert b.get_job(rec["job_id"])["status"] == "succeeded"


@needs_db
def test_worker_failure_is_recorded_not_raised(monkeypatch, jobs):
    fake_worker(monkeypatch, "import sys; sys.stderr.write('model exploded'); sys.exit(3)")
    rec = b.benchmark_embedding_model(ALLOWED, "bench_tiny")
    assert rec["status"] == "failed" and "model exploded" in rec["error"]


@needs_db
def test_timeout_is_recorded_and_process_killed(monkeypatch, jobs):
    cfg = config.settings
    monkeypatch.setattr(cfg.benchmarks, "max_runtime_seconds", 1)
    fake_worker(monkeypatch, "import time; time.sleep(60)")
    rec = b.benchmark_embedding_model(ALLOWED, "bench_tiny")
    assert rec["status"] == "timeout" and "1" in rec["error"]


@needs_db
def test_garbage_output_is_a_failure(monkeypatch, jobs):
    fake_worker(monkeypatch, "print('not json')")
    assert b.benchmark_embedding_model(ALLOWED, "bench_tiny")["status"] == "failed"


def test_worker_command_never_enables_remote_code():
    cmd = b._worker_cmd(ALLOWED, "rev", TINY, 0)
    assert "trust_remote_code" not in " ".join(cmd) and cmd[1:3] == ["-m", "tools.benchmarks"]


# ---- real worker, real model, reproducibility ------------------------------

@needs_db
def test_real_run_is_reproducible_within_tolerance(monkeypatch, jobs):
    monkeypatch.setattr(b.hf_client, "get_model_details", _REAL_DETAILS)  # real pinned revision
    r1 = b.benchmark_embedding_model(ALLOWED, "bench_tiny", seed=1)
    r2 = b.benchmark_embedding_model(ALLOWED, "bench_tiny", seed=1)
    assert r1["status"] == r2["status"] == "succeeded", (r1.get("error"), r2.get("error"))
    for k in ("recall@1", "recall@5", "mrr@10", "ndcg@10"):
        assert r1["metrics"][k] == pytest.approx(r2["metrics"][k], abs=1e-6)
    assert r1["metrics"]["recall@5"] >= 0.8  # sanity: MiniLM solves the tiny topical fixture
    assert {"platform", "cpu_count", "torch", "sentence_transformers"} <= set(r1["hardware"])
    assert r1["metrics"]["n_queries"] == 6
