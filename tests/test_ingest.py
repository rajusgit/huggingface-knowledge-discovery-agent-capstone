import psycopg
import pytest
from pgvector.psycopg import register_vector

from src import config, ingest

SRC = ingest.Source(
    source_id="test-src", repo="o/r", path="docs/a.md", ref="v1",
    canonical_url="https://example.org/a", owner="o", license="Apache-2.0",
    permitted_use="index_text", store_text=True, category="transformers",
    attribution="o",
)
TEXT = "# Title\n\nFirst paragraph.\n\n## Part\n\nSecond paragraph."


def fake_embed(texts):
    return [[0.1] * 384 for _ in texts]


def test_chunk_id_is_deterministic_and_content_sensitive():
    a = ingest.chunk_id("s", "rev", "hierarchical", 0, "h1", corpus_rev="c")
    assert a == ingest.chunk_id("s", "rev", "hierarchical", 0, "h1", corpus_rev="c")
    assert a != ingest.chunk_id("s", "rev", "hierarchical", 0, "h2", corpus_rev="c")
    assert len(a) == 64


def test_corpus_revision_is_order_independent():
    assert ingest.corpus_revision([("a", "1"), ("b", "2")]) == ingest.corpus_revision([("b", "2"), ("a", "1")])


@pytest.mark.parametrize("use", ["pending_review", "rejected"])
def test_unapproved_sources_refused(use):
    bad = ingest.Source(**{**SRC.__dict__, "permitted_use": use})
    with pytest.raises(ingest.IngestRefused):
        ingest.check_permitted(bad)


def test_load_sources_reads_registry():
    srcs = ingest.load_sources()
    assert srcs and all(s.permitted_use in ("index_text", "reference_only", "pending_review", "rejected") for s in srcs.values())


@pytest.fixture
def conn():
    try:
        c = psycopg.connect(config.settings.dsn, connect_timeout=15,
                            options=f"-c search_path={config.settings.database.schema_name},public")
    except Exception as e:  # no DB available
        pytest.skip(f"database unavailable: {e}")
    register_vector(c)
    yield c
    c.rollback()  # integration tests never persist rows
    c.close()


def _run(conn, src=SRC, strategy="hierarchical"):
    ingest.upsert_source(conn, src, version="v1")
    run_id = ingest.start_run(conn, "test-rev", strategy)
    n = ingest.ingest_source(conn, src, TEXT, "sha123", run_id, fake_embed, strategy)
    return run_id, n


def test_ingest_inserts_provenance_and_is_idempotent(conn):
    run_id, n = _run(conn)
    assert n > 0
    row = conn.execute(
        "SELECT source_url, heading_path, revision, license, content_hash, content FROM doc_chunks "
        "WHERE run_id=%s ORDER BY chunk_index LIMIT 1", (run_id,)).fetchone()
    assert row[0] == SRC.canonical_url and row[2] == "sha123" and row[3] == "Apache-2.0"
    assert row[4].startswith("sha256:") and row[5]
    _, n2 = _run(conn)
    assert n2 == 0


def test_reference_only_stores_no_text(conn):
    src = ingest.Source(**{**SRC.__dict__, "source_id": "test-ref", "permitted_use": "reference_only", "store_text": False})
    run_id, n = _run(conn, src)
    assert n > 0
    assert conn.execute("SELECT count(*) FROM doc_chunks WHERE run_id=%s AND content IS NOT NULL", (run_id,)).fetchone()[0] == 0


def test_publish_activates_and_supersedes(conn):
    r1 = ingest.start_run(conn, "rev-a", "hierarchical")
    ingest.publish(conn, r1, require_chunks=False)
    r2 = ingest.start_run(conn, "rev-b", "hierarchical")
    ingest.publish(conn, r2, require_chunks=False)
    st = dict(conn.execute("SELECT run_id, status FROM ingest_runs WHERE run_id IN (%s,%s)", (r1, r2)).fetchall())
    assert st[r2] == "active" and st[r1] == "superseded"


def test_publish_refuses_empty_run(conn):
    r = ingest.start_run(conn, "rev-empty", "hierarchical")
    with pytest.raises(ingest.IngestRefused):
        ingest.publish(conn, r)


# ---- batch ingestion: reuse, validation, withdraw ---------------------------------------

def counting_embed():
    calls = []

    def embed(texts):
        calls.append(len(texts))
        return [[0.1] * 384 for _ in texts]
    embed.calls = calls
    return embed


def test_pinned_sha_needs_no_network(monkeypatch):
    monkeypatch.setattr(ingest, "_get", lambda *a, **k: pytest.fail("network used"))
    s = ingest.Source(**{**SRC.__dict__, "ref": "v9", "sha": "deadbeef"})
    assert ingest.resolve_revision(s) == ("v9", "deadbeef")


def test_rerun_embeds_nothing(conn):
    emb = counting_embed()
    ingest.upsert_source(conn, SRC, version="v1")
    run_id = ingest.start_run(conn, "reuse-rev", "hierarchical")
    n1 = ingest.ingest_source(conn, SRC, TEXT, "sha123", run_id, emb, "hierarchical")
    first_calls = sum(emb.calls)
    n2 = ingest.ingest_source(conn, SRC, TEXT, "sha123", run_id, emb, "hierarchical")
    assert n1 > 0 and n2 == 0 and sum(emb.calls) == first_calls      # second pass: no embedding at all


def test_new_revision_reuses_embeddings_for_unchanged_chunks(conn):
    emb = counting_embed()
    ingest.upsert_source(conn, SRC, version="v1")
    r1 = ingest.start_run(conn, "rev-one", "hierarchical")
    ingest.ingest_source(conn, SRC, TEXT, "sha-one", r1, emb, "hierarchical")
    before = sum(emb.calls)
    r2 = ingest.start_run(conn, "rev-two", "hierarchical")
    stats = {}
    n = ingest.ingest_source(conn, SRC, TEXT, "sha-two", r2, emb, "hierarchical", stats=stats)
    assert n > 0 and sum(emb.calls) == before                          # new run rows, zero new embeddings
    assert stats["reused"] == n and stats["embedded"] == 0


def test_validate_run_passes_on_good_run(conn):
    run_id, _ = _run(conn)
    res = ingest.validate_run(conn, run_id, check_urls=lambda urls: [], search=lambda q, run_id: [{"source_url": "https://example.org/a"}],
                              smoke=[{"query": "q", "expect": "example.org"}])
    assert res["ok"] and res["errors"] == [] and res["stats"]["chunks"] > 0 and res["stats"]["sources"] == 1


def test_validate_run_flags_each_problem(conn):
    empty = ingest.start_run(conn, "val-empty", "hierarchical")
    assert "no chunks" in " ".join(ingest.validate_run(conn, empty, check_urls=lambda u: [], search=lambda q, run_id: [], smoke=[])["errors"])
    run_id, _ = _run(conn)
    bad_url = ingest.validate_run(conn, run_id, check_urls=lambda urls: list(urls), search=lambda q, run_id: [], smoke=[])
    assert not bad_url["ok"] and "broken" in " ".join(bad_url["errors"])
    miss = ingest.validate_run(conn, run_id, check_urls=lambda u: [], search=lambda q, run_id: [{"source_url": "https://other"}],
                               smoke=[{"query": "q", "expect": "example.org"}])
    assert not miss["ok"] and "smoke" in " ".join(miss["errors"])


def test_failed_validation_leaves_previous_active_run(conn):
    a = ingest.start_run(conn, "keep-a", "hierarchical")
    ingest.publish(conn, a, require_chunks=False)
    b = ingest.start_run(conn, "bad-b", "hierarchical")           # no chunks -> invalid
    res = ingest.validate_run(conn, b, check_urls=lambda u: [], search=lambda q, run_id: [], smoke=[])
    ingest.finish_validation(conn, b, res)
    st = dict(conn.execute("SELECT run_id, status FROM ingest_runs WHERE run_id IN (%s,%s)", (a, b)).fetchall())
    assert st[a] == "active" and st[b] == "failed"
    with pytest.raises(ingest.IngestRefused):
        ingest.publish(conn, b)


def test_withdraw_tombstones_chunks_and_rejects_source(conn):
    run_id, _ = _run(conn)
    n = ingest.withdraw_source(conn, SRC.source_id)
    assert n > 0
    assert conn.execute("SELECT count(*) FROM doc_chunks WHERE run_id=%s AND deleted_at IS NULL", (run_id,)).fetchone()[0] == 0
    assert conn.execute("SELECT permitted_use FROM sources WHERE source_id=%s", (SRC.source_id,)).fetchone()[0] == "rejected"


# ---- URL checking classifies by outcome --------------------------------------------------

def test_url_check_only_definitive_failures_are_broken(monkeypatch):
    statuses = {"ok": 200, "redirect": 301, "gone": 404, "removed": 410, "rate": 429, "oops": 503, "net": None}
    monkeypatch.setattr(ingest, "_url_status", lambda url: statuses[url])
    monkeypatch.setattr(ingest, "_sleep", lambda s: None)
    broken, unverified = ingest.check_urls_detail(list(statuses))
    assert sorted(broken) == ["gone", "removed"]
    assert sorted(unverified) == ["net", "oops", "rate"]          # not counted as broken


def test_url_check_retries_transient_failures_before_giving_up(monkeypatch):
    seq = iter([429, 429, 200])
    monkeypatch.setattr(ingest, "_url_status", lambda url: next(seq))
    monkeypatch.setattr(ingest, "_sleep", lambda s: None)
    assert ingest.check_urls_detail(["u"]) == ([], [])


def test_each_strategy_gets_its_own_run_over_the_same_corpus(conn):
    strategies = ("hierarchical", "fixed50", "recursive200", "hybrid128")
    ingest.upsert_source(conn, SRC, version="v1")
    runs = {}
    for s in strategies:
        runs[s] = ingest.start_run(conn, "same-corpus", s)
        assert ingest.ingest_source(conn, SRC, TEXT, "sha123", runs[s], fake_embed, s) > 0
    assert len(set(runs.values())) == 4
    got = dict(conn.execute("SELECT chunk_strategy, run_id FROM ingest_runs WHERE corpus_revision='same-corpus'").fetchall())
    assert got == runs
    # chunk ids differ per strategy, so one corpus can hold all four side by side
    assert conn.execute("SELECT count(DISTINCT chunk_strategy) FROM doc_chunks WHERE run_id = ANY(%s)", (list(runs.values()),)).fetchone()[0] == 4


@pytest.mark.parametrize("flag,is_default,expected", [
    (None, True, True),       # the configured default chunker is published
    (None, False, False),     # experiment variants are validated but never become active
    (True, False, True),      # unless explicitly asked
    (False, True, False),
])
def test_publish_policy(flag, is_default, expected):
    default = config.settings.retrieval.chunker             # chosen by the variant experiment, so read it
    other = next(c for c in ("hierarchical", "fixed50", "recursive200", "hybrid128") if c != default)
    assert ingest.should_publish(default if is_default else other, flag) is expected


def test_two_runs_over_the_same_revision_each_hold_the_full_corpus(conn):
    """Regression: chunk_id used to ignore the run, so a second run at an unchanged repo commit
    silently inserted nothing (ON CONFLICT DO NOTHING) and was missing its pages."""
    emb = counting_embed()
    ingest.upsert_source(conn, SRC, version="v1")
    a = ingest.start_run(conn, "corpus-A", "hierarchical")
    b = ingest.start_run(conn, "corpus-B", "hierarchical")           # different corpus_revision, SAME source revision
    na = ingest.ingest_source(conn, SRC, TEXT, "same-sha", a, emb, "hierarchical")
    nb = ingest.ingest_source(conn, SRC, TEXT, "same-sha", b, emb, "hierarchical")
    assert na > 0 and nb == na
    for run in (a, b):
        assert conn.execute("SELECT count(*) FROM doc_chunks WHERE run_id=%s", (run,)).fetchone()[0] == na
    ids = conn.execute("SELECT count(DISTINCT chunk_id) FROM doc_chunks WHERE run_id IN (%s,%s)", (a, b)).fetchone()[0]
    assert ids == 2 * na                                              # ids are unique per run
    assert ingest.ingest_source(conn, SRC, TEXT, "same-sha", b, emb, "hierarchical") == 0   # still idempotent within a run


def test_chunk_id_depends_on_the_corpus_revision():
    assert ingest.chunk_id("s", "rev", "hierarchical", 0, "h", corpus_rev="A") != ingest.chunk_id("s", "rev", "hierarchical", 0, "h", corpus_rev="B")
