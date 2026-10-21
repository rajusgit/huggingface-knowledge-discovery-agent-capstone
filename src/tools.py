"""The three tools: search_knowledge, get_table_schema, validate_sql.
Phase 1 implements only search_knowledge. Phases 4–5 add the others.
"""
from src.db import pool
from src.embed import embed


def search_knowledge(query: str, doc_type: str | None = None, k: int = 5) -> list[dict]:
    """Retrieve top-k chunks by cosine similarity to the query.

    Args:
        query: question to embed and search for
        doc_type: filter by 'qa', 'schemas', or None for all
        k: number of results to return

    Returns:
        list of dicts with keys: source, chunk_idx, title, content, similarity
    """
    from pgvector.psycopg import register_vector

    q_embedding = embed([query])[0]
    pool.open()
    with pool.connection() as conn:
        register_vector(conn)
        params = [q_embedding, q_embedding, k]
        where_clause = ""
        if doc_type:
            where_clause = " AND doc_type = %s"
            params.insert(2, doc_type)
        sql = f"""
            SELECT source, chunk_idx, title, content,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM slackbot.doc_chunks
            WHERE 1=1 {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        rows = conn.execute(sql, params).fetchall()
    return [
        {
            "source": r[0],
            "chunk_idx": r[1],
            "title": r[2],
            "content": r[3],
            "similarity": float(r[4]),
        }
        for r in rows
    ]


def get_table_schema(table_fqn: str) -> dict:
    """Get schema for a table. Phase 5: reads live Unity Catalog.
    For now: JSON fixtures in input/schemas/.
    """
    import json
    from pathlib import Path
    from src.config import settings

    # Extract table name (main.aws_logs.s3_inventory -> s3_inventory)
    name = table_fqn.split(".")[-1]
    schema_file = settings.input_dir / "schemas" / f"{name}.json"
    if not schema_file.exists():
        raise ValueError(f"no schema for {table_fqn}")
    return json.loads(schema_file.read_text())


def validate_sql(sql: str, table_fqn: str) -> tuple[bool, str]:
    """Validate SQL: SELECT-only, row/cost limits, columns exist.
    Phase 5 function. For now: basic checks only.
    """
    import sqlglot

    try:
        tree = sqlglot.parse_one(sql, dialect="databricks")
    except Exception as e:
        return False, f"parse error: {e}"

    if not isinstance(tree, sqlglot.exp.Select):
        return False, "not a SELECT"

    # Stub: Phase 5 adds column validation against the schema.
    return True, "ok"
