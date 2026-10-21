"""Connection pool + schema init.

    .venv/bin/python -m src.db --init    apply schema.sql, then report what exists
"""
import sys
from pathlib import Path

from psycopg_pool import ConnectionPool

from src.config import settings

SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"

# Pooled because every Slack event opens a connection from a laptop to RDS over
# the internet -- a TLS handshake per message is latency we can just not pay.
pool = ConnectionPool(
    settings.dsn, min_size=1, max_size=settings.database.pool_size, open=False,
    kwargs={"options": "-c search_path=slackbot,public"},
)


def init() -> None:
    pool.open()
    with pool.connection() as conn:
        conn.execute(SCHEMA_SQL.read_text())
    print(f"applied {SCHEMA_SQL.name}")


def status() -> None:
    pool.open()
    with pool.connection() as conn:
        ver = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        assert ver, "pgvector extension is not installed -- schema init did not run"
        print(f"pgvector {ver[0]}")
        rows = conn.execute("""
            SELECT c.relname,
                   (SELECT count(*) FROM pg_index i WHERE i.indrelid = c.oid) AS indexes,
                   pg_total_relation_size(c.oid) AS bytes
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'slackbot' AND c.relkind = 'r'
            ORDER BY c.relname
        """).fetchall()
        assert rows, "no tables in schema slackbot"
        for name, idx, size in rows:
            n = conn.execute(f"SELECT count(*) FROM slackbot.{name}").fetchone()[0]
            print(f"  slackbot.{name:<18} rows={n:<6} indexes={idx} bytes={size}")


if __name__ == "__main__":
    if "--init" in sys.argv:
        init()
    status()
