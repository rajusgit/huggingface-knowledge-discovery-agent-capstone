import psycopg
import pytest
from pgvector import Vector
from pgvector.psycopg import register_vector

from src import config, ingest, retrieval


def vec(i):  # unit-ish vectors pointing at different axes
    v = [0.0] * 384
    v[i] = 1.0
    return v


def embed_q(i):
    return lambda texts: [vec(i) for _ in texts]


@pytest.fixture
def conn():
    try:
        c = psycopg.connect(config.settings.dsn, connect_timeout=15,
                            options=f"-c search_path={config.settings.database.schema_name},public")
    except Exception as e:
        pytest.skip(f"database unavailable: {e}")
    register_vector(c)
    yield c
    c.rollback()
    c.close()


def seed(conn):
    src = ingest.Source("rt-src", "o/r", "a.md", "v1", "https://example.org/rt", "o", "MIT",
                        "index_text", True, "catA", "o")
    ingest.upsert_source(conn, src, "v1")
    run = ingest.start_run(conn, "rt-rev", "hierarchical")
    for i, cat in ((1, "catA"), (2, "catB"), (3, "catA")):
        conn.execute(
            """INSERT INTO doc_chunks (chunk_id, run_id, source_id, source_url, heading_path, revision,
               license, category, chunk_strategy, chunk_index, content, content_hash, embedding)
               VALUES (%s,%s,'rt-src','https://example.org/rt',%s,'rev','MIT',%s,'hierarchical',%s,%s,%s,%s)""",
            (f"rt-{i}", run, ["Title", f"S{i}"], cat, i, f"content {i}", f"sha256:{i}", Vector(vec(i))))
    return run


def test_returns_nearest_with_provenance(conn):
    run = seed(conn)
    hits = retrieval.search_knowledge("q", run_id=run, k=3, embed_fn=embed_q(1), conn=conn, threshold=0.5)
    assert [h["chunk_id"] for h in hits] == ["rt-1"]
    h = hits[0]
    assert h["source_url"] == "https://example.org/rt" and h["heading_path"] == ["Title", "S1"]
    assert h["revision"] == "rev" and h["content"] == "content 1" and h["score"] > 0.99


def test_threshold_drops_weak_matches(conn):
    run = seed(conn)
    assert retrieval.search_knowledge("q", run_id=run, embed_fn=embed_q(9), conn=conn, threshold=0.3) == []


def test_category_filter(conn):
    run = seed(conn)
    hits = retrieval.search_knowledge("q", run_id=run, category="catB", embed_fn=embed_q(1), conn=conn, threshold=-1)
    assert {h["chunk_id"] for h in hits} == {"rt-2"}


def test_deleted_rows_excluded(conn):
    run = seed(conn)
    conn.execute("UPDATE doc_chunks SET deleted_at=now() WHERE chunk_id='rt-1'")
    hits = retrieval.search_knowledge("q", run_id=run, embed_fn=embed_q(1), conn=conn, threshold=0.5)
    assert hits == []


def test_default_scope_is_active_run_only(conn):
    run = seed(conn)  # run is 'running', not active
    hits = retrieval.search_knowledge("q", embed_fn=embed_q(1), conn=conn, threshold=0.9)
    assert "rt-1" not in {h["chunk_id"] for h in hits}
    ingest.publish(conn, run)
    hits = retrieval.search_knowledge("q", embed_fn=embed_q(1), conn=conn, threshold=0.9)
    assert [h["chunk_id"] for h in hits] == ["rt-1"]


def test_real_corpus_smoke():
    try:
        hits = retrieval.search_knowledge("How do I limit the length of generated text?")
    except Exception as e:
        pytest.skip(f"database unavailable: {e}")
    if not hits:
        pytest.skip("no active run / nothing above threshold")
    assert hits[0]["source_url"].startswith("https://huggingface.co/docs/")


# ---- T18: hybrid retrieval (pgvector + Postgres full-text, RRF) and reranking --------------------

def seed_hybrid(conn):
    """4 chunks. Query vector = axis 0. Dense order: a > b > c > target. Only `target` contains the rare term."""
    src = ingest.Source("hy-src", "o/r", "a.md", "v1", "https://example.org/hy", "o", "MIT", "index_text", True, "catA", "o")
    ingest.upsert_source(conn, src, "v1")
    run = ingest.start_run(conn, "hy-rev", "hierarchical")
    rows = [("hy-a", "alpha generic text about models", [1.0, 0.0]),
            ("hy-b", "beta generic text about training", [0.9, 0.4]),
            ("hy-c", "gamma generic text about data", [0.6, 0.8]),
            ("hy-target", "the zyxwvutsrq parameter controls output", [0.2, 0.98])]
    for i, (cid, content, (x, y)) in enumerate(rows):
        v = [0.0] * 384
        v[0], v[1] = x, y
        conn.execute(
            """INSERT INTO doc_chunks (chunk_id, run_id, source_id, source_url, heading_path, revision, license,
               category, chunk_strategy, chunk_index, content, content_hash, embedding)
               VALUES (%s,%s,'hy-src','https://example.org/hy',%s,'rev','MIT','catA','hierarchical',%s,%s,%s,%s)""",
            (cid, run, ["T", cid], i, content, f"sha256:{cid}", Vector(v)))
    return run


def test_hybrid_ranks_keyword_match_above_dense_only(conn):
    run = seed_hybrid(conn)
    kw = dict(run_id=run, k=4, embed_fn=embed_q(0), conn=conn, threshold=-1)
    dense = [h["chunk_id"] for h in retrieval.search_knowledge("zyxwvutsrq parameter", mode="dense", **kw)]
    hybrid = [h["chunk_id"] for h in retrieval.search_knowledge("zyxwvutsrq parameter", mode="hybrid", **kw)]
    assert dense.index("hy-target") == 3                       # dense buries the exact-term chunk
    assert hybrid.index("hy-target") < dense.index("hy-target")
    assert hybrid[0] == "hy-target" or hybrid[1] == "hy-target"


def test_hybrid_keeps_lexical_matches_below_the_score_threshold(conn):
    run = seed_hybrid(conn)
    hits = retrieval.search_knowledge("zyxwvutsrq parameter", mode="hybrid", run_id=run, k=4,
                                      embed_fn=embed_q(0), conn=conn, threshold=0.7)
    ids = {h["chunk_id"] for h in hits}
    assert "hy-target" in ids and "hy-c" not in ids            # cosine(target)=0.2<0.7 but it matched the keyword
    assert next(h for h in hits if h["chunk_id"] == "hy-target")["lexical"] is True
    assert all("rrf" in h for h in hits)


def test_hybrid_respects_scope_filters(conn):
    run = seed_hybrid(conn)
    assert retrieval.search_knowledge("zyxwvutsrq", mode="hybrid", run_id=run, category="catB", embed_fn=embed_q(0),
                                      conn=conn, threshold=-1) == []
    conn.execute("UPDATE doc_chunks SET deleted_at=now() WHERE chunk_id='hy-target'")
    ids = {h["chunk_id"] for h in retrieval.search_knowledge("zyxwvutsrq", mode="hybrid", run_id=run, embed_fn=embed_q(0),
                                                             conn=conn, threshold=-1)}
    assert "hy-target" not in ids


def test_unknown_mode_is_rejected(conn):
    with pytest.raises(ValueError, match="mode"):
        retrieval.search_knowledge("q", mode="semantic", embed_fn=embed_q(0), conn=conn)


class FakeScorer:
    """cross-encoder stand-in: score = how many query words appear in the passage."""

    def __call__(self, pairs):
        return [float(sum(w in p.lower() for w in q.lower().split())) for q, p in pairs]


def test_retrieve_reranks_candidates_and_returns_top_n(conn, monkeypatch):
    run = seed_hybrid(conn)
    base = dict(run_id=run, k=4, embed_fn=embed_q(0), conn=conn, threshold=-1, top_n=2, mode="dense")
    plain = retrieval.retrieve("zyxwvutsrq parameter", rerank=False, **base)
    assert [h["chunk_id"] for h in plain] == ["hy-a", "hy-b"]                   # retrieval order, cut to top_n
    reranked = retrieval.retrieve("zyxwvutsrq parameter", rerank=True, scorer=FakeScorer(), **base)
    assert reranked[0]["chunk_id"] == "hy-target" and len(reranked) == 2
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]
    assert {h["chunk_id"] for h in reranked} <= {"hy-a", "hy-b", "hy-c", "hy-target"}  # same run, only order/cut changes


def test_retrieve_defaults_come_from_config(conn, monkeypatch):
    run = seed_hybrid(conn)
    cfg = retrieval.config.settings.retrieval
    monkeypatch.setattr(cfg, "rerank", False)
    monkeypatch.setattr(cfg, "top_n", 1)
    monkeypatch.setattr(cfg, "mode", "dense")
    hits = retrieval.retrieve("q", run_id=run, k=4, embed_fn=embed_q(0), conn=conn, threshold=-1)
    assert len(hits) == 1 and hits[0]["chunk_id"] == "hy-a"


def test_rerank_module_real_model_orders_relevant_first(monkeypatch):
    from src import rerank
    monkeypatch.setattr(retrieval.config.settings.retrieval, "reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    monkeypatch.setattr(rerank, "_model", None)
    try:
        scores = rerank.score("how do I limit the generated length?",
                              ["Set max_new_tokens to control how many tokens are generated.",
                               "The Eiffel Tower is located in Paris, France."])
    except Exception as e:
        pytest.skip(f"cross-encoder unavailable: {e}")
    assert scores[0] > scores[1]


def test_reranker_refuses_nan_scores_instead_of_silently_keeping_the_order():
    """Regression: CPU BLAS returned NaN for mmap'd weights; sorting by NaN left the order unchanged and
    every 'rerank' variant silently behaved like 'no rerank'."""
    from src import rerank
    with pytest.raises(RuntimeError, match="NaN"):
        rerank.score("q", ["a", "b"], scorer=lambda pairs: [float("nan")] * len(pairs))
