"""registry -> fetch (pinned commit) -> parse/chunk -> embed -> insert -> publish.

    uv run python standalone-scripts/ingest.py --source <source_id>
"""
import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml
from pgvector import Vector

from src import config
from src.chunking import chunk_markdown

APPROVED = ("index_text", "reference_only")


class IngestRefused(RuntimeError):
    pass


@dataclass
class Source:
    source_id: str
    repo: str
    path: str
    ref: str
    canonical_url: str
    owner: str
    license: str
    permitted_use: str
    store_text: bool
    category: str
    attribution: str


def load_sources(path: Path | None = None) -> dict[str, Source]:
    path = path or config.ROOT / config.settings.ingest.sources_file
    items = yaml.safe_load(path.read_text())["sources"]
    return {i["source_id"]: Source(**i) for i in items}


def check_permitted(src: Source) -> None:
    if src.permitted_use not in APPROVED:
        raise IngestRefused(f"{src.source_id}: permitted_use={src.permitted_use!r}; needs review before indexing")


def chunk_id(source_id: str, revision: str, strategy: str, index: int, content_hash: str) -> str:
    return hashlib.sha256(f"{source_id}|{revision}|{strategy}|{index}|{content_hash}".encode()).hexdigest()


def corpus_revision(pairs: list[tuple[str, str]]) -> str:
    joined = "\n".join(f"{s}@{r}" for s, r in sorted(pairs))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


# --- network ---------------------------------------------------------------

def _get(url: str, accept: str | None = None) -> str:
    cfg = config.settings.ingest
    headers = {"User-Agent": "hf-knowledge-agent-ingest"}
    if accept:
        headers["Accept"] = accept
    for attempt in range(cfg.fetch_max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=cfg.fetch_timeout_seconds) as r:
                return r.read().decode()
        except Exception:
            if attempt == cfg.fetch_max_retries:
                raise
            time.sleep(2**attempt)


def resolve_revision(src: Source) -> tuple[str, str]:
    """-> (version tag, commit sha). ref 'latest' = latest GitHub release tag."""
    api = config.settings.ingest.github_api
    tag = src.ref
    if tag == "latest":
        tag = json.loads(_get(f"{api}/repos/{src.repo}/releases/latest"))["tag_name"]
    sha = _get(f"{api}/repos/{src.repo}/commits/{tag}", accept="application/vnd.github.sha").strip()
    return tag, sha


def fetch_text(src: Source, sha: str) -> str:
    return _get(f"{config.settings.ingest.raw_base}/{src.repo}/{sha}/{src.path}")


# --- database --------------------------------------------------------------

def upsert_source(conn, src: Source, version: str) -> None:
    conn.execute(
        """INSERT INTO sources (source_id, canonical_url, owner, license, permitted_use, store_text,
                                version, attribution, category)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (source_id) DO UPDATE SET canonical_url=EXCLUDED.canonical_url,
             owner=EXCLUDED.owner, license=EXCLUDED.license, permitted_use=EXCLUDED.permitted_use,
             store_text=EXCLUDED.store_text, version=EXCLUDED.version,
             attribution=EXCLUDED.attribution, category=EXCLUDED.category, updated_at=now()""",
        (src.source_id, src.canonical_url, src.owner, src.license, src.permitted_use,
         src.store_text, version, src.attribution, src.category),
    )


def start_run(conn, corpus_rev: str, strategy: str) -> int:
    """Get-or-create the run for (corpus_revision, chunk_strategy)."""
    conn.execute(
        """INSERT INTO ingest_runs (corpus_revision, chunk_strategy, embedding_model, status)
           VALUES (%s,%s,%s,'running') ON CONFLICT (corpus_revision, chunk_strategy) DO NOTHING""",
        (corpus_rev, strategy, config.settings.embedding.model),
    )
    return conn.execute(
        "SELECT run_id FROM ingest_runs WHERE corpus_revision=%s AND chunk_strategy=%s",
        (corpus_rev, strategy),
    ).fetchone()[0]


def ingest_source(conn, src: Source, text: str, revision: str, run_id: int, embed_fn, strategy: str) -> int:
    """Chunk, embed and insert one source. Returns the number of NEW rows (0 on re-run)."""
    check_permitted(src)
    chunks = chunk_markdown(text, strategy)
    batch = config.settings.ingest.embed_batch_size
    vectors = []
    for i in range(0, len(chunks), batch):
        vectors += embed_fn([c.chunk_text for c in chunks[i:i + batch]])
    inserted = 0
    for idx, (c, vec) in enumerate(zip(chunks, vectors)):
        h = "sha256:" + hashlib.sha256(c.content.encode()).hexdigest()
        cur = conn.execute(
            """INSERT INTO doc_chunks (chunk_id, run_id, source_id, source_url, heading_path, revision,
                   license, category, access, chunk_strategy, chunk_index, content, content_hash, embedding)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'public',%s,%s,%s,%s,%s)
               ON CONFLICT (chunk_id) DO NOTHING""",
            (chunk_id(src.source_id, revision, strategy, idx, h), run_id, src.source_id,
             src.canonical_url, c.heading_path, revision, src.license, src.category, strategy, idx,
             c.content if src.store_text else None, h, Vector(vec)),
        )
        inserted += cur.rowcount
    return inserted


def publish(conn, run_id: int, require_chunks: bool = True) -> None:
    """Make run_id the single active run (previous active -> superseded), atomically."""
    if require_chunks:
        n = conn.execute("SELECT count(*) FROM doc_chunks WHERE run_id=%s", (run_id,)).fetchone()[0]
        if n == 0:
            raise IngestRefused(f"run {run_id} has no chunks; refusing to publish")
    conn.execute("UPDATE ingest_runs SET status='superseded' WHERE status='active' AND run_id<>%s", (run_id,))
    conn.execute("UPDATE ingest_runs SET status='active', finished_at=now() WHERE run_id=%s", (run_id,))


# --- orchestration ---------------------------------------------------------

def run(source_ids: list[str], strategy: str | None = None) -> None:
    from pgvector.psycopg import register_vector

    from src.db import get_pool
    from src.embed import embed

    strategy = strategy or config.settings.retrieval.chunker
    sources = load_sources()
    chosen = [sources[s] for s in source_ids]
    for s in chosen:
        check_permitted(s)
    resolved = {s.source_id: resolve_revision(s) for s in chosen}
    rev = corpus_revision([(sid, sha) for sid, (_, sha) in resolved.items()])
    pool = get_pool()
    pool.open()
    with pool.connection() as conn:
        register_vector(conn)
        run_id = start_run(conn, rev, strategy)
        total = 0
        for s in chosen:
            tag, sha = resolved[s.source_id]
            upsert_source(conn, s, tag)
            total += ingest_source(conn, s, fetch_text(s, sha), sha, run_id, embed, strategy)
        publish(conn, run_id)
    print(f"run {run_id} (corpus {rev}, {strategy}): {total} new chunks, published")
