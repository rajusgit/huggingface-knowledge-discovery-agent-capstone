"""Dense retrieval over hf_discovery.doc_chunks (pgvector cosine).

Scope = the active ingest run unless run_id is given (experiments query a specific run).
"""
from pgvector import Vector
from pgvector.psycopg import register_vector

from src import config


def search_knowledge(
    query: str,
    *,
    category: str | None = None,
    k: int | None = None,
    run_id: int | None = None,
    threshold: float | None = None,
    embed_fn=None,
    conn=None,
) -> list[dict]:
    """Top-k chunks by cosine similarity, dropping those below the score threshold.

    Returns dicts: chunk_id, source_id, source_url, heading_path, revision, content, score.
    """
    cfg = config.settings.retrieval
    k = k or cfg.k
    threshold = cfg.score_threshold if threshold is None else threshold
    if embed_fn is None:
        from src.embed import embed as embed_fn
    q = Vector(embed_fn([query])[0])

    where, params = ["c.deleted_at IS NULL"], {"q": q, "k": k}
    if run_id is None:
        where.append("r.status = 'active'")
    else:
        where.append("c.run_id = %(run_id)s")
        params["run_id"] = run_id
    if category:
        where.append("c.category = %(category)s")
        params["category"] = category
    sql = f"""
        SELECT c.chunk_id, c.source_id, c.source_url, c.heading_path, c.revision, c.content,
               1 - (c.embedding <=> %(q)s) AS score
        FROM doc_chunks c JOIN ingest_runs r USING (run_id)
        WHERE {' AND '.join(where)}
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(k)s"""

    def run(c):
        register_vector(c)
        return c.execute(sql, params).fetchall()

    if conn is not None:
        rows = run(conn)
    else:
        from src.db import get_pool
        pool = get_pool()
        pool.open()
        with pool.connection() as c:
            rows = run(c)
    keys = ("chunk_id", "source_id", "source_url", "heading_path", "revision", "content", "score")
    hits = [dict(zip(keys, r)) for r in rows]
    for h in hits:
        h["score"] = float(h["score"])
    return [h for h in hits if h["score"] >= threshold]
