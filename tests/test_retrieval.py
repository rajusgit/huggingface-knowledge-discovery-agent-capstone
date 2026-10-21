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
