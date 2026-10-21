import psycopg
import pytest
from pgvector.psycopg import register_vector

from src import config, ingest
from src.chunking import Chunk

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
    a = ingest.chunk_id("s", "rev", "hierarchical", 0, "h1")
    assert a == ingest.chunk_id("s", "rev", "hierarchical", 0, "h1")
    assert a != ingest.chunk_id("s", "rev", "hierarchical", 0, "h2")
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
