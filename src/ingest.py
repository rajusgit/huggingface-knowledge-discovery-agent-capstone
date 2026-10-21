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
    sha: str | None = None   # pinned commit; None = resolve `ref` at ingest time


def load_sources(path: Path | None = None) -> dict[str, Source]:
    path = path or config.ROOT / config.settings.ingest.sources_file
    items = yaml.safe_load(path.read_text())["sources"]
    fields = set(Source.__dataclass_fields__)
    return {i["source_id"]: Source(**{k: v for k, v in i.items() if k in fields}) for i in items}


def check_permitted(src: Source) -> None:
    if src.permitted_use not in APPROVED:
        raise IngestRefused(f"{src.source_id}: permitted_use={src.permitted_use!r}; needs review before indexing")


def chunk_id(source_id: str, revision: str, strategy: str, index: int, content_hash: str, corpus_rev: str = "") -> str:
    """Deterministic and unique per run: the corpus revision is part of the key, so two runs that
    cover the same page at the same repo commit each get their own rows."""
    return hashlib.sha256(f"{corpus_rev}|{source_id}|{revision}|{strategy}|{index}|{content_hash}".encode()).hexdigest()


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
    """-> (version tag, commit sha). A pinned sha needs no network; ref 'latest' = latest release tag."""
    if src.sha:
        return src.ref, src.sha
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


def _as_vector(v) -> Vector:
    return v if isinstance(v, Vector) else Vector(v)


def _reusable_embeddings(conn, src: Source, strategy: str, hashes: list[str]) -> dict:
    """(content_hash, heading_path) -> embedding already stored by any earlier run of the same model."""
    rows = conn.execute(
        """SELECT DISTINCT ON (c.content_hash, c.heading_path) c.content_hash, c.heading_path, c.embedding
           FROM doc_chunks c JOIN ingest_runs r USING (run_id)
           WHERE c.source_id=%s AND c.chunk_strategy=%s AND r.embedding_model=%s AND c.content_hash = ANY(%s)""",
        (src.source_id, strategy, config.settings.embedding.model, hashes)).fetchall()
    return {(h, tuple(p)): e for h, p, e in rows}


def ingest_source(conn, src: Source, text: str, revision: str, run_id: int, embed_fn, strategy: str,
                  stats: dict | None = None) -> int:
    """Chunk, embed and insert one source. Returns the number of NEW rows (0 on re-run).

    Only chunks that aren't already in this run are embedded, and embeddings of chunks that are
    unchanged since an earlier run are reused. stats (optional) collects {embedded, reused}.
    """
    check_permitted(src)
    corpus_rev = conn.execute("SELECT corpus_revision FROM ingest_runs WHERE run_id=%s", (run_id,)).fetchone()[0]
    stats = stats if stats is not None else {}
    stats.setdefault("embedded", 0)
    stats.setdefault("reused", 0)
    items = []
    for idx, c in enumerate(chunk_markdown(text, strategy)):
        h = "sha256:" + hashlib.sha256(c.content.encode()).hexdigest()
        items.append((idx, c, h, chunk_id(src.source_id, revision, strategy, idx, h, corpus_rev)))
    have = {r[0] for r in conn.execute("SELECT chunk_id FROM doc_chunks WHERE run_id=%s AND source_id=%s",
                                       (run_id, src.source_id)).fetchall()}
    todo = [it for it in items if it[3] not in have]
    if not todo:
        return 0
    reuse = _reusable_embeddings(conn, src, strategy, [it[2] for it in todo])
    vectors: dict[int, object] = {}
    missing = []
    for idx, c, h, _ in todo:
        hit = reuse.get((h, tuple(c.heading_path)))
        if hit is not None:
            vectors[idx] = hit
            stats["reused"] += 1
        else:
            missing.append((idx, c))
    batch = config.settings.ingest.embed_batch_size
    for i in range(0, len(missing), batch):
        part = missing[i:i + batch]
        for (idx, _), vec in zip(part, embed_fn([c.chunk_text for _, c in part])):
            vectors[idx] = vec
            stats["embedded"] += 1
    rows = [(cid, run_id, src.source_id, src.canonical_url, c.heading_path, revision, src.license, src.category,
             strategy, idx, c.content if src.store_text else None, h, _as_vector(vectors[idx]))
            for idx, c, h, cid in todo]
    cur = conn.cursor()
    # one pipelined round trip per batch instead of one per row (~26x faster against a remote RDS)
    cur.executemany(
        """INSERT INTO doc_chunks (chunk_id, run_id, source_id, source_url, heading_path, revision,
               license, category, access, chunk_strategy, chunk_index, content, content_hash, embedding)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'public',%s,%s,%s,%s,%s)
           ON CONFLICT (chunk_id) DO NOTHING""", rows)
    return cur.rowcount


# --- validation / publishing / withdrawal -------------------------------------

_sleep = time.sleep  # patched in tests
DEFINITELY_BROKEN = {404, 410}


def _url_status(url: str) -> int | None:
    """HTTP status of a URL (HEAD, falling back to GET); None if unreachable."""
    import urllib.error

    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method, headers={"User-Agent": "hf-knowledge-agent-ingest"})
            with urllib.request.urlopen(req, timeout=config.settings.ingest.url_check_timeout_seconds) as r:
                return r.status
        except urllib.error.HTTPError as e:
            if method == "GET" or e.code not in (403, 405, 501):
                return e.code
        except Exception:
            return None
    return None


def check_urls_detail(urls: list[str]) -> tuple[list[str], list[str]]:
    """-> (broken, unverified). Only a definitive 404/410 is *broken*. Rate limits (429), server errors
    and network failures are retried with backoff, then reported as *unverified*, never as broken."""
    from concurrent.futures import ThreadPoolExecutor

    cfg = config.settings.ingest

    def check(url: str) -> str:
        for attempt in range(cfg.fetch_max_retries + 1):
            status = _url_status(url)
            if status is not None and (200 <= status < 400):
                return "ok"
            if status in DEFINITELY_BROKEN:
                return "broken"
            if attempt < cfg.fetch_max_retries:
                _sleep(cfg.url_check_backoff_seconds * 2**attempt)
        return "unverified"

    def paced(url: str) -> str:
        _sleep(cfg.url_check_delay_seconds)
        return check(url)

    with ThreadPoolExecutor(cfg.url_check_workers) as ex:
        results = list(zip(urls, ex.map(paced, urls)))
    return [u for u, r in results if r == "broken"], [u for u, r in results if r == "unverified"]


def check_urls_http(urls: list[str]) -> list[str]:
    """-> the URLs that are definitively broken (unverified ones are warned about, not failed)."""
    broken, unverified = check_urls_detail(urls)
    if unverified:
        print(f"  warning: {len(unverified)} URLs could not be verified (rate-limited/unreachable), e.g. {unverified[:3]}")
    return broken


def validate_run(conn, run_id: int, *, check_urls=None, search=None, smoke: list[dict] | None = None) -> dict:
    """Checks before a run may be published: chunks exist, few duplicates, citation URLs resolve,
    and smoke queries retrieve what they should. -> {ok, errors, stats}."""
    from src import retrieval

    cfg = config.settings.ingest
    check_urls = check_urls or check_urls_http
    search = search or (lambda q, run_id: retrieval.search_knowledge(q, run_id=run_id))
    smoke = cfg.smoke_queries if smoke is None else smoke
    errors: list[str] = []
    chunks, sources, distinct = conn.execute(
        "SELECT count(*), count(DISTINCT source_id), count(DISTINCT content_hash) FROM doc_chunks WHERE run_id=%s AND deleted_at IS NULL",
        (run_id,)).fetchone()
    stats = {"chunks": chunks, "sources": sources}
    if chunks == 0:
        return {"ok": False, "errors": ["no chunks in this run"], "stats": stats}
    stats["duplicate_ratio"] = round(1 - distinct / chunks, 4)
    if stats["duplicate_ratio"] > cfg.max_duplicate_ratio:
        errors.append(f"duplicate content ratio {stats['duplicate_ratio']} exceeds {cfg.max_duplicate_ratio}")
    urls = [r[0] for r in conn.execute(
        "SELECT DISTINCT source_url FROM doc_chunks WHERE run_id=%s AND deleted_at IS NULL", (run_id,)).fetchall()]
    broken = check_urls(urls)
    stats["broken_urls"] = broken[:20]
    if len(broken) / len(urls) > cfg.max_broken_url_ratio:
        errors.append(f"{len(broken)}/{len(urls)} citation URLs are broken, e.g. {broken[:3]}")
    failed = [s["query"] for s in smoke
              if not any(s["expect"] in h["source_url"] for h in search(s["query"], run_id=run_id))]
    stats["smoke_failed"] = failed
    if failed:
        errors.append(f"smoke queries did not retrieve the expected source: {failed}")
    return {"ok": not errors, "errors": errors, "stats": stats}


def finish_validation(conn, run_id: int, result: dict) -> None:
    from psycopg.types.json import Jsonb

    conn.execute("UPDATE ingest_runs SET status=%s, stats=stats || %s WHERE run_id=%s AND status<>'active'",
                 ("validated" if result["ok"] else "failed", Jsonb({**result["stats"], "errors": result["errors"]}), run_id))


def withdraw_source(conn, source_id: str) -> int:
    """Pull a source out of circulation: reject it and tombstone its chunks in every run."""
    conn.execute("UPDATE sources SET permitted_use='rejected', updated_at=now() WHERE source_id=%s", (source_id,))
    return conn.execute("UPDATE doc_chunks SET deleted_at=now() WHERE source_id=%s AND deleted_at IS NULL",
                        (source_id,)).rowcount


def publish(conn, run_id: int, require_chunks: bool = True) -> None:
    """Make run_id the single active run (previous active -> superseded), atomically."""
    if conn.execute("SELECT status FROM ingest_runs WHERE run_id=%s", (run_id,)).fetchone()[0] == "failed":
        raise IngestRefused(f"run {run_id} failed validation; refusing to publish")
    if require_chunks:
        n = conn.execute("SELECT count(*) FROM doc_chunks WHERE run_id=%s", (run_id,)).fetchone()[0]
        if n == 0:
            raise IngestRefused(f"run {run_id} has no chunks; refusing to publish")
    conn.execute("UPDATE ingest_runs SET status='superseded' WHERE status='active' AND run_id<>%s", (run_id,))
    conn.execute("UPDATE ingest_runs SET status='active', finished_at=now() WHERE run_id=%s", (run_id,))


# --- orchestration ---------------------------------------------------------

def should_publish(strategy: str, flag: bool | None) -> bool:
    """Only the configured default chunker is published automatically; A/B variants are validated
    and kept (queryable by run_id) without replacing the active run."""
    return strategy == config.settings.retrieval.chunker if flag is None else flag


def run(source_ids: list[str] | None = None, strategy: str | None = None, publish_run: bool | None = None) -> int:
    """Ingest the given sources (default: every approved source in the registry), validate, publish.

    Commits after each source so an interrupted run resumes cheaply; the run only becomes `active`
    after validation passes. Returns the run id.
    """
    from pgvector.psycopg import register_vector

    from src.db import get_pool
    from src.embed import embed

    strategy = strategy or config.settings.retrieval.chunker
    sources = load_sources()
    chosen = [sources[s] for s in source_ids] if source_ids else [s for s in sources.values() if s.permitted_use in APPROVED]
    for s in chosen:
        check_permitted(s)
    resolved, tags = {}, {}
    for s in chosen:
        key = (s.repo, s.ref, s.sha)
        tags.setdefault(key, resolve_revision(s))  # one lookup per repo+ref, not per page
        resolved[s.source_id] = tags[key]
    rev = corpus_revision([(sid, sha) for sid, (_, sha) in resolved.items()])
    pool = get_pool()
    pool.open()
    totals = {"embedded": 0, "reused": 0}
    with pool.connection() as conn:
        register_vector(conn)
        run_id = start_run(conn, rev, strategy)
        conn.commit()
        new = 0
        for i, s in enumerate(chosen, 1):
            tag, sha = resolved[s.source_id]
            upsert_source(conn, s, tag)
            try:
                text = fetch_text(s, sha)
            except Exception as e:
                print(f"[{i}/{len(chosen)}] SKIP {s.source_id}: fetch failed ({e})")
                continue
            n = ingest_source(conn, s, text, sha, run_id, embed, strategy, stats=totals)
            conn.commit()
            new += n
            print(f"[{i}/{len(chosen)}] {s.source_id}: +{n} chunks")
        # A/B strategies cover the same URLs as the main run; don't hammer huggingface.co again for them
        same_urls_as_main = strategy != config.settings.retrieval.chunker
        result = validate_run(conn, run_id, check_urls=(lambda urls: []) if same_urls_as_main else None)
        result["stats"].update(totals)
        finish_validation(conn, run_id, result)
        published = result["ok"] and should_publish(strategy, publish_run)
        if published:
            publish(conn, run_id)
        conn.commit()
    print(f"run {run_id} (corpus {rev}, {strategy}): {new} new chunks, embedded {totals['embedded']}, "
          f"reused {totals['reused']}; {'PUBLISHED' if published else 'validated, not published' if result['ok'] else 'FAILED validation'}")
    for e in result["errors"]:
        print("  validation error:", e)
    return run_id
