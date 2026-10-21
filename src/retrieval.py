"""Retrieval over hf_discovery.doc_chunks.

search_knowledge()  one retrieval pass: mode 'dense' (pgvector cosine) or 'hybrid'
                    (dense + Postgres full-text, fused with reciprocal-rank fusion).
retrieve()          the pipeline: search_knowledge -> optional cross-encoder rerank -> top_n.

Scope = the active ingest run unless run_id is given (experiments query a specific run).
"""
from pgvector import Vector
from pgvector.psycopg import register_vector

from src import config

MODES = ("dense", "hybrid")
_COLS = "c.chunk_id, c.source_id, c.source_url, c.heading_path, c.revision, c.content"
_KEYS = ("chunk_id", "source_id", "source_url", "heading_path", "revision", "content", "score")


def _scope(run_id, category, params):
    where = ["c.deleted_at IS NULL"]
    if run_id is None:
        where.append("r.status = 'active'")
    else:
        where.append("c.run_id = %(run_id)s")
        params["run_id"] = run_id
    if category:
        where.append("c.category = %(category)s")
        params["category"] = category
    return " AND ".join(where)


def search_knowledge(
    query: str,
    *,
    category: str | None = None,
    k: int | None = None,
    run_id: int | None = None,
    threshold: float | None = None,
    mode: str | None = None,
    embed_fn=None,
    conn=None,
) -> list[dict]:
    """Top-k chunks. Returns dicts: chunk_id, source_id, source_url, heading_path, revision, content,
    score (cosine similarity); hybrid mode also adds rrf and lexical (matched the keyword query).

    Dense drops chunks below the score threshold. Hybrid keeps a chunk if it clears the threshold
    OR matched the keywords, so exact-term hits aren't discarded for low cosine similarity.
    """
    cfg = config.settings.retrieval
    mode = mode or cfg.mode
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    k = k or cfg.k
    threshold = cfg.score_threshold if threshold is None else threshold
    if embed_fn is None:
        from src.embed import embed as embed_fn
    params: dict = {"q": Vector(embed_fn([query])[0]), "k": k}
    where = _scope(run_id, category, params)

    if mode == "dense":
        sql = f"""
            SELECT {_COLS}, 1 - (c.embedding <=> %(q)s) AS score
            FROM doc_chunks c JOIN ingest_runs r USING (run_id)
            WHERE {where}
            ORDER BY c.embedding <=> %(q)s
            LIMIT %(k)s"""
    else:
        params.update(qtext=query, rrf_k=cfg.rrf_k)
        sql = f"""
            WITH scope AS (
                SELECT {_COLS}, c.embedding, c.tsv
                FROM doc_chunks c JOIN ingest_runs r USING (run_id) WHERE {where}),
            dense AS (
                SELECT chunk_id, row_number() OVER (ORDER BY embedding <=> %(q)s) AS rnk
                FROM scope ORDER BY embedding <=> %(q)s LIMIT %(k)s),
            lex AS (
                SELECT s.chunk_id, row_number() OVER (ORDER BY ts_rank_cd(s.tsv, tq.q) DESC) AS rnk
                FROM scope s, websearch_to_tsquery('english', %(qtext)s) AS tq(q)
                WHERE s.tsv @@ tq.q ORDER BY ts_rank_cd(s.tsv, tq.q) DESC LIMIT %(k)s),
            fused AS (
                SELECT chunk_id, sum(1.0 / (%(rrf_k)s + rnk)) AS rrf
                FROM (SELECT * FROM dense UNION ALL SELECT * FROM lex) u GROUP BY chunk_id)
            SELECT s.chunk_id, s.source_id, s.source_url, s.heading_path, s.revision, s.content,
                   1 - (s.embedding <=> %(q)s) AS score, f.rrf,
                   EXISTS (SELECT 1 FROM lex WHERE lex.chunk_id = s.chunk_id) AS lexical
            FROM fused f JOIN scope s USING (chunk_id)
            ORDER BY f.rrf DESC, s.chunk_id
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
    hits = []
    for r in rows:
        h = dict(zip(_KEYS, r))
        h["score"] = float(h["score"])
        if mode == "hybrid":
            h["rrf"], h["lexical"] = float(r[7]), bool(r[8])
        hits.append(h)
    return [h for h in hits if h["score"] >= threshold or h.get("lexical")]


def retrieve(query: str, *, mode: str | None = None, rerank: bool | None = None, top_n: int | None = None,
             scorer=None, **kw) -> list[dict]:
    """The retrieval pipeline: candidates (k) -> optional cross-encoder rerank -> top_n.

    rerank=None follows config (retrieval.rerank). Without reranking the candidates
    keep their retrieval order and are just cut to top_n, so on/off differ only in ordering.
    """
    cfg = config.settings.retrieval
    top_n = top_n or cfg.top_n
    use_rerank = cfg.rerank if rerank is None else rerank
    hits = search_knowledge(query, mode=mode, **kw)
    if use_rerank and hits:
        from src import rerank as reranker

        for h, s in zip(hits, reranker.score(query, [h["content"] for h in hits], scorer=scorer)):
            h["rerank_score"] = float(s)
        hits = sorted(hits, key=lambda h: -h["rerank_score"])
    return hits[:top_n]
