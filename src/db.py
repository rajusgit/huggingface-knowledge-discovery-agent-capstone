"""Connection pool + schema init for the hf_discovery schema.

    uv run python -m src.db --init    apply schema.sql, then report what exists
    uv run python -m src.db           report only
"""
import sys
from pathlib import Path

from psycopg_pool import ConnectionPool

from src import config

SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"
VECTOR_MARKER = "-- VECTOR_INDEX"
_HNSW = ("CREATE INDEX IF NOT EXISTS doc_chunks_embedding_hnsw\n"
         "    ON hf_discovery.doc_chunks USING hnsw (embedding vector_cosine_ops);")
# ponytail: lists=100 is a guess; tune once the corpus size is known
_IVFFLAT = ("CREATE INDEX IF NOT EXISTS doc_chunks_embedding_ivfflat\n"
            "    ON hf_discovery.doc_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")

_pool: ConnectionPool | None = None


def search_path() -> str:
    return f"{config.settings.database.schema_name},public"


def get_pool() -> ConnectionPool:
    # Pooled because each request would otherwise pay a TLS handshake to RDS.
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            config.settings.dsn, min_size=1, max_size=config.settings.database.pool_size, open=False,
            kwargs={"options": f"-c search_path={search_path()}"},
        )
    return _pool


def __getattr__(name: str):
    if name == "pool":  # back-compat for `from src.db import pool`
        return get_pool()
    raise AttributeError(name)


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split(".")[:3] if p.isdigit())


def render_schema(pgvector_version: str) -> str:
    index = _HNSW if _version_tuple(pgvector_version) >= (0, 5, 0) else _IVFFLAT
    return SCHEMA_SQL.read_text().replace(VECTOR_MARKER, index)


def init() -> None:
    pool = get_pool()
    pool.open()
    with pool.connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        ver = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        assert ver, "pgvector extension could not be created"
        conn.execute(render_schema(ver[0]))
    print(f"applied {SCHEMA_SQL.name} (pgvector {ver[0]})")


def status() -> None:
    schema = config.settings.database.schema_name
    pool = get_pool()
    pool.open()
    with pool.connection() as conn:
        ver = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        assert ver, "pgvector extension is not installed -- run with --init"
        print(f"pgvector {ver[0]}")
        rows = conn.execute("""
            SELECT c.relname,
                   (SELECT count(*) FROM pg_index i WHERE i.indrelid = c.oid) AS indexes,
                   pg_total_relation_size(c.oid) AS bytes
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relkind = 'r'
            ORDER BY c.relname
        """, (schema,)).fetchall()
        assert rows, f"no tables in schema {schema}"
        for name, idx, size in rows:
            n = conn.execute(f'SELECT count(*) FROM "{schema}"."{name}"').fetchone()[0]
            print(f"  {schema}.{name:<18} rows={n:<6} indexes={idx} bytes={size}")


if __name__ == "__main__":
    if "--init" in sys.argv:
        init()
    status()
