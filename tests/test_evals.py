import json

import pytest
import yaml
from fastapi.testclient import TestClient

from observability import evals
from src import api, config, llm, prompts, retrieval

V = evals.Variant("hierarchical", "dense", False)


def chunk(sid, n=0, content="text"):
    return {"chunk_id": f"{sid}-{n}", "source_id": sid, "source_url": f"https://x/{sid}", "heading_path": ["T"],
            "revision": "r", "content": content, "score": 0.9 - n * 0.1}


# ---- variants ---------------------------------------------------------------------------------

def test_sixteen_unique_variants_with_stable_names():
    vs = evals.all_variants()
    assert len(vs) == 16 and len({v.name for v in vs}) == 16
    assert {v.chunker for v in vs} == {"hierarchical", "fixed50", "recursive200", "hybrid128"}
    assert {v.mode for v in vs} == {"dense", "hybrid"} and {v.rerank for v in vs} == {True, False}
    assert V.name == "hierarchical|dense|norerank" and evals.Variant("fixed50", "hybrid", True).name == "fixed50|hybrid|rerank"


# ---- deterministic evaluators -------------------------------------------------------------

def test_recall_and_mrr_use_expected_source_ids():
    out = {"retrieved": [chunk("a"), chunk("b"), chunk("c")]}
    assert evals.recall_at_k(out, {"source_ids": ["c"]}, 5) == 1.0
    assert evals.recall_at_k(out, {"source_ids": ["c"]}, 2) == 0.0
    assert evals.recall_at_k(out, {"source_ids": ["a", "z"]}, 5) == 0.5
    assert evals.mrr(out, {"source_ids": ["b"]}) == 0.5 and evals.mrr(out, {"source_ids": ["z"]}) == 0.0


def test_citation_correctness_requires_an_expected_source_and_no_invalid_numbers():
    ok = {"citations": [chunk("a")], "invalid_citations": []}
    assert evals.citation_correct(ok, {"source_ids": ["a"]}) == 1.0
    assert evals.citation_correct({"citations": [chunk("b")], "invalid_citations": []}, {"source_ids": ["a"]}) == 0.0
    assert evals.citation_correct({"citations": [], "invalid_citations": []}, {"source_ids": ["a"]}) == 0.0
    assert evals.citation_correct({"citations": [chunk("a")], "invalid_citations": [9]}, {"source_ids": ["a"]}) == 0.0


def test_keyword_coverage_and_refusal():
    assert evals.keyword_coverage({"answer": "Set MAX_NEW_TOKENS to 50"}, {"keywords": ["max_new_tokens", "beam"]}) == 0.5
    assert evals.correct_refusal({"answer": prompts.NO_CONTEXT_ANSWER, "retrieved": [], "citations": []}, {"should_refuse": True}) == 1.0
    assert evals.correct_refusal({"answer": "Canberra [1]", "retrieved": [chunk("a")], "citations": [chunk("a")]}, {"should_refuse": True}) == 0.0
    assert evals.correct_refusal({"answer": "I cannot answer that from the provided documentation.", "retrieved": [chunk("a")], "citations": []}, {"should_refuse": True}) == 1.0


# ---- LLM judges go through src.llm ------------------------------------------------------------

def fake_judge(monkeypatch, replies):
    calls = []

    def complete(messages, **kw):
        calls.append(messages)
        return {"content": replies.pop(0), "tool_calls": [], "usage": {"cost": 0.001}}
    monkeypatch.setattr(llm, "complete", complete)
    return calls


def test_faithfulness_is_supported_claims_over_total(monkeypatch):
    calls = fake_judge(monkeypatch, ['```json\n{"supported": 3, "total": 4}\n```'])
    out = {"answer": "a [1]", "contexts": [chunk("a")]}
    assert evals.faithfulness(out, {"question": "q"}) == 0.75
    assert len(calls) == 1 and any("a" in m["content"] for m in calls[0])        # judged via src.llm.complete


def test_judges_return_none_on_garbage_instead_of_crashing(monkeypatch):
    fake_judge(monkeypatch, ["I think it's good", '{"score": 7}'])
    out = {"answer": "x", "contexts": [chunk("a")]}
    assert evals.faithfulness(out, {"question": "q"}) is None
    assert evals.answer_relevancy(out, {"question": "q"}) is None                  # out-of-range score rejected


def test_context_precision_and_recall(monkeypatch):
    fake_judge(monkeypatch, ['{"relevant": [true, false, true]}', '{"score": 1}'])
    out = {"answer": "x", "contexts": [chunk("a"), chunk("b"), chunk("c")]}
    ex = {"question": "q", "reference_answer": "ref"}
    assert evals.context_precision(out, ex) == pytest.approx(2 / 3)
    assert evals.context_recall(out, ex) == 1.0


# ---- the pipeline under test ---------------------------------------------------------------------

def test_answer_with_variant_scopes_to_the_chunkers_run_and_limits_context(monkeypatch):
    seen = {}

    def fake_retrieve(q, **kw):
        seen.update(kw)
        return [chunk("a", 0), chunk("b", 1), chunk("c", 2), chunk("d", 3), chunk("e", 4)]
    monkeypatch.setattr(retrieval, "retrieve", fake_retrieve)
    monkeypatch.setattr(llm, "complete", lambda m, **k: {"content": "Answer [1][2].", "usage": {"cost": 0.002}, "tool_calls": []})
    v = evals.Variant("fixed50", "hybrid", True)
    out = evals.answer_with_variant("q?", v, {"fixed50": 77, "hierarchical": 1})
    assert seen["run_id"] == 77 and seen["mode"] == "hybrid" and seen["rerank"] is True and seen["top_n"] == 5
    assert len(out["retrieved"]) == 5 and len(out["contexts"]) == config.settings.retrieval.top_n
    assert [c["n"] for c in out["citations"]] == [1, 2] and out["cost"] == 0.002 and out["latency_ms"] >= 0


def test_answer_with_variant_refuses_without_context_and_skips_the_llm(monkeypatch):
    monkeypatch.setattr(retrieval, "retrieve", lambda q, **k: [])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: pytest.fail("no LLM without context"))
    out = evals.answer_with_variant("q?", V, {"hierarchical": 1})
    assert out["answer"] == prompts.NO_CONTEXT_ANSWER and out["citations"] == [] and out["cost"] == 0.0


# ---- aggregation + selection --------------------------------------------------------------------

def test_summary_has_mean_ci_and_ignores_missing_scores():
    s = evals.summarize([1.0, 0.0, 1.0, None, 1.0, 1.0, 0.0, 1.0])
    assert s["n"] == 7 and s["mean"] == pytest.approx(5 / 7) and s["ci_low"] <= s["mean"] <= s["ci_high"]
    assert evals.summarize([1.0, 0.0, 1.0, 1.0]) == evals.summarize([1.0, 0.0, 1.0, 1.0])   # seeded bootstrap
    assert evals.summarize([None, None]) == {"n": 0, "mean": None, "ci_low": None, "ci_high": None}


def test_percentiles():
    assert evals.percentile([10, 20, 30, 40, 50], 50) == 30
    assert evals.percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48)


def row(name, q, cost=0.01, p95=100):
    return {"variant": name, "composite": q, "cost_per_query": cost, "latency_p95_ms": p95}


def test_winner_is_confirmed_on_heldout_and_ties_break_on_cost():
    dev = [row("a", .90), row("b", .88), row("c", .87), row("d", .60)]
    held = {"a": row("a", .70), "b": row("b", .80, cost=.02), "c": row("c", .80, cost=.01), "d": row("d", .99)}
    win = evals.select_winner(dev, held, top=3)
    assert win["variant"] == "c"          # d never reaches heldout; b and c tie, c is cheaper


def test_apply_winner_rewrites_only_the_selection_keys(tmp_path):
    src = (config.ROOT / "config" / "config.yaml").read_text()
    p = tmp_path / "config.yaml"
    p.write_text(src)
    evals.apply_winner(evals.Variant("fixed50", "hybrid", True), "exp-123", path=p)
    new = p.read_text()
    cfg = yaml.safe_load(new)["retrieval"]
    assert (cfg["chunker"], cfg["mode"], cfg["rerank"]) == ("fixed50", "hybrid", True) and "exp-123" in new
    evals.apply_winner(evals.Variant("hierarchical", "dense", False), "exp-124", path=p)
    after = yaml.safe_load(p.read_text())
    assert after["retrieval"]["rerank"] is False
    assert after["retrieval"]["reranker"] == yaml.safe_load(src)["retrieval"]["reranker"]    # model name is never lost
    assert p.read_text().count("eval-winner") == 1 and "exp-124" in p.read_text()             # marker replaced, not duplicated
    a, b = yaml.safe_load(src), after
    for k in a:
        if k != "retrieval":
            assert a[k] == b[k]
    for k in a["retrieval"]:
        if k not in ("chunker", "mode", "rerank"):
            assert a["retrieval"][k] == b["retrieval"][k]


# ---- whole experiment with fakes + API ---------------------------------------------------------------

def golden(n_doc=4):
    items = []
    for i in range(n_doc):
        items.append({"id": f"doc_qa-{i}", "category": "doc_qa", "question": f"q{i}", "split": "dev" if i < n_doc - 1 else "heldout",
                      "reference_answer": "ref", "expected": {"source_ids": ["a"], "keywords": ["text"]}})
    items.append({"id": "un-1", "category": "unanswerable", "question": "weather?", "split": "dev", "reference_answer": None,
                  "expected": {"should_refuse": True}})
    return items


def test_run_experiment_scores_every_variant_on_both_splits(monkeypatch):
    def fake_answer(q, v, runs, **kw):
        good = v.mode == "hybrid"
        ctx = [chunk("a" if good else "z")]
        return {"answer": "text [1]", "retrieved": ctx, "contexts": ctx, "citations": [{**ctx[0], "n": 1}],
                "invalid_citations": [], "latency_ms": 10, "cost": 0.001}
    monkeypatch.setattr(evals, "answer_with_variant", fake_answer)
    monkeypatch.setattr(evals, "faithfulness", lambda o, e: 1.0)
    monkeypatch.setattr(evals, "answer_relevancy", lambda o, e: 1.0)
    monkeypatch.setattr(evals, "context_precision", lambda o, e: 1.0)
    monkeypatch.setattr(evals, "context_recall", lambda o, e: 1.0)
    timed = []

    def fake_timer(v, items, runs):
        timed.append(v.name)
        return [40.0, 60.0] if v.rerank else [10.0, 20.0]
    res = evals.run_experiment(golden(), {c: 1 for c in ("hierarchical", "fixed50", "recursive200", "hybrid128")}, workers=2, timer=fake_timer)
    assert len(timed) == 16                                                   # a sequential timing pass per variant
    rr = next(v for v in res["variants"] if v["variant"] == "fixed50|dense|rerank")
    nr = next(v for v in res["variants"] if v["variant"] == "fixed50|dense|norerank")
    assert rr["retrieval_p95_ms"] > nr["retrieval_p95_ms"] and nr["retrieval_p50_ms"] == 15.0
    assert len(res["variants"]) == 16 and res["experiment_id"].startswith("exp-")
    hy = next(v for v in res["variants"] if v["variant"] == "hierarchical|hybrid|rerank")
    de = next(v for v in res["variants"] if v["variant"] == "hierarchical|dense|rerank")
    assert hy["dev"]["recall@5"]["mean"] == 1.0 and de["dev"]["recall@5"]["mean"] == 0.0
    assert hy["heldout"]["recall@5"]["n"] == 1 and hy["latency_p95_ms"] == 10 and hy["cost_per_query"] > 0
    assert res["winner"]["variant"].split("|")[1] == "hybrid"


def test_api_serves_saved_leaderboard(monkeypatch, tmp_path):
    p = tmp_path / "variants.json"
    p.write_text(json.dumps({"experiment_id": "exp-1", "variants": [{"variant": "x"}], "winner": {"variant": "x"}}))
    monkeypatch.setattr(config.settings.evals, "results_file", str(p))
    c = TestClient(api.app)
    assert c.get("/api/evals/variants").json()["experiment_id"] == "exp-1"
    monkeypatch.setattr(config.settings.evals, "results_file", str(tmp_path / "missing.json"))
    r = c.get("/api/evals/variants")
    assert r.status_code == 200 and r.json()["variants"] == [] and "not been run" in r.json()["message"]
