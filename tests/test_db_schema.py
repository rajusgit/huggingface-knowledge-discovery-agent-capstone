import re

import pytest

from src import db
from src.config import ROOT

TABLES = ["sources", "ingest_runs", "doc_chunks", "sessions", "messages", "benchmark_runs"]


@pytest.fixture(autouse=True)
def pw(monkeypatch):
    monkeypatch.setenv("TICKERPULSE_DB_PASSWORD", "pw")


def test_schema_declares_all_tables_in_new_schema():
    sql = (ROOT / "src" / "schema.sql").read_text()
    for t in TABLES:
        assert f"CREATE TABLE IF NOT EXISTS hf_discovery.{t} " in sql, t
    assert "slackbot" not in sql


def test_schema_is_idempotent():
    sql = (ROOT / "src" / "schema.sql").read_text()
    for stmt in re.findall(r"CREATE (?:UNIQUE )?(?:TABLE|INDEX|SCHEMA|EXTENSION)[^;]*", sql):
        assert "IF NOT EXISTS" in stmt, stmt[:70]


def test_render_schema_hnsw_when_pgvector_ge_0_5():
    assert "USING hnsw" in db.render_schema("0.7.4")
    assert "USING hnsw" in db.render_schema("0.5.0")
    assert db.VECTOR_MARKER not in db.render_schema("0.7.4")


def test_render_schema_ivfflat_fallback_when_old():
    sql = db.render_schema("0.4.4")
    assert "USING ivfflat" in sql and "hnsw" not in sql


def test_search_path_uses_configured_schema():
    assert db.search_path() == "hf_discovery,public"


def test_pool_is_lazy_and_not_opened_on_import():
    assert db.get_pool().closed
