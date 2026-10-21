# Hugging Face Knowledge & Discovery Agent — Technical Architecture

**Status:** Capstone implementation blueprint · **Scope:** Publicly available, license-permitted AI engineering knowledge + live Hugging Face discovery · **Primary journey:** Learn → Discover → Compare → Evaluate

## 1. Product scope and design principles

Build a React chat-and-discovery application for aspiring AI/ML engineers. Users ask implementation questions grounded in public technical documentation; discover Hugging Face models, embedding models and datasets; compare suitability against hardware, task, language and license constraints; and run **small, reproducible** model benchmarks. Every factual answer links to source documents or current API records; experimental results are clearly distinguished from publisher-reported metrics.

**Design principles (developer-confirmed):**

1. **Modular backend in flat `src/*.py` modules** — one responsibility per file, explicit imports, no framework-style subpackages (§9).
2. **LLM access through OpenRouter only**, via a self-contained `src/llm.py` that imports nothing from other `src` modules (§3.1).
3. **Single LLM gateway:** every LLM call in the project — the agent's, and those made by NeMo Guardrails, RAGAS and openevals through thin adapters — goes through `src/llm.py`. Same OpenRouter model everywhere.
4. **Everything tunable lives in `config/config.yaml`** (models, embedding/router encoders, thresholds, k/top_n, chunk sizes, timeouts, retries, TTLs, benchmark limits). No model name or tunable is hard-coded in Python.
5. **Reuse this repo's components** — `config/config.yaml`, `src/config.py`, `src/embed.py` (all-MiniLM-L6-v2) and the `src/db.py` Postgres/pgvector pool. Postgres + pgvector replaces Qdrant. This project gets its **own new schema `hf_discovery`** on the existing RDS instance; no existing tables are reused (§5.3).

**Non-goals:** No wholesale Hugging Face mirror, no private/gated corpus, no bulk model weights or datasets in the vector store, no automatic execution of untrusted repository code, and no claim that popularity proves model quality. Public access is **not** equivalent to permission to persist or redistribute full text: verify licenses and terms per source before ingestion.

## 2. End-to-end architecture

```text
                        ┌──────────────────────────────┐
                        │ React + TypeScript Dashboard │
                        │ Chat | Search | Compare | Eval│
                        └──────────────┬───────────────┘
                                       │ HTTPS / SSE streaming
                        ┌──────────────▼───────────────┐
                        │ FastAPI (src/api.py)          │
                        │ auth, sessions, input guards  │
                        └──────────────┬───────────────┘
                                       │
                        ┌──────────────▼───────────────┐
                        │ LangGraph (agents/graph.py)  │
                        │ learn | discover | compare    │
                        │ evaluate | combined           │
                        └───────────┬───────────┬──────┘
                                    │           │
                  ┌─────────────────▼──┐   ┌────▼────────────────┐
                  │ RAG (src/retrieval.py)│   │ MCP client          │
                  │ pgvector + PG FTS  │   │ MultiServerMCPClient│
                  │ rerank + cite      │   └────┬────────────────┘
                  └─────────────────┬──┘        │
                                    │      ┌────▼────────────────┐
     Public docs / permitted cards  │      │ FastMCP server      │
        │                           │      │ (tools/mcp_server.py) │
  license allowlist (sources)       │      │ HF client, bench    │
        │                           │      └────┬────────────────┘
 Docling / Markdown parser          │           │ REST / isolated jobs
        │                           │      Hugging Face Hub
 chunk + embed (src/embed.py)       │      Models / Datasets / Cards
        │                           │
        └──► PostgreSQL + pgvector ◄┘
             schema hf_discovery (RDS)
                                    │
                         ┌──────────▼───────────┐
                         │ LLM synthesis        │
                         │ src/llm.py → OpenRouter│
                         └──────────┬───────────┘
                                    ▼
                       Answer | comparisons | benchmark report

 Cross-cutting: PostgreSQL hf_discovery (chunks, sources, sessions, runs)
 In-process caches (LLM exact-match, HF TTL) · LangSmith (traces/evals)
 OpenRouter (LLM, usage/cost) · PG checkpointer (thread memory)
```

**Routing:** Documentation questions → RAG; current model/dataset search → MCP REST calls; structured filtering → API parameters / SQL `WHERE` filters, not vector similarity alone; comparison → MCP facts + RAG limitations; benchmarks → controlled tool execution. An answer can combine both branches, with separate source attribution.

## 3. Complete technology stack

| Layer | Default stack | Role / implementation notes | Alternative / decision |
|---|---|---|---|
| Web UI | React, TypeScript, Vite, Tailwind CSS | Chat, citations, filter controls, compare table, benchmark results, pipeline variant toggle | Next.js if SSR/auth needs grow |
| API | Python 3.11+, FastAPI, Pydantic v2, Uvicorn | Typed REST endpoints, streaming responses, validation, auth and rate limiting | None needed for MVP |
| Agent | LangGraph (`agents/`), hand-built nodes | Explicit intent routing, state, tool calls, conditional workflows. Nodes call `src/llm.py` for every LLM request. **Postgres checkpointer** (`hf_discovery`) keyed by `thread_id` for short-term memory | `create_agent` for simpler flows |
| MCP | Python FastMCP server (`tools/`); `MultiServerMCPClient` | Typed tool contracts, source API wrappers; local stdio for MVP | Remote HTTP transport later; verify chosen MCP SDK support (SSE may be legacy) |
| External data | `huggingface_hub` and Hugging Face REST API | Live discovery, repository metadata, file listings and selected card fetches | Direct `httpx` REST calls for unsupported endpoints |
| Parsing | Docling for PDF; Markdown/HTML parsers; CSV rows as documents if used | Preserve sections, headings, code blocks, source positions | Regex → OCR → VLM only when source scans require it |
| Chunking | HierarchicalChunker + heading breadcrumbs | Section-aware source citations | fixed 50 words; recursive 200 chars/20 overlap; Docling HybridChunker token max 128 |
| Embeddings | `all-MiniLM-L6-v2` via existing `src/embed.py` (sentence-transformers, 384-dim, loaded once, `HF_HUB_OFFLINE=1`) | Same module for ingest and query so vectors never drift; record model revision | BGE-small; compare on retrieval quality |
| Vector store | PostgreSQL + pgvector, schema `hf_discovery` on existing RDS, via `src/db.py` pool | `vector(384)` column, cosine distance (`<=>`), HNSW index | — (replaces Qdrant) |
| Keyword search | Postgres full-text search (`tsvector` + `ts_rank_cd`, GIN index) | Hybrid lexical + dense via reciprocal-rank fusion in SQL | Dense-only baseline (replaces rank-bm25) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (from config) | Rerank top 10 candidates → top 3 | No reranker baseline |
| LLM | OpenRouter (`https://openrouter.ai/api/v1`) via `openai` SDK in `src/llm.py`; model `google/gemini-3.5-flash-lite` | Single model for routing and synthesis; 3 retries with backoff; usage/cost from OpenRouter response | LiteLLM and LangChain chat models are not used |
| Operational DB | Same PostgreSQL schema `hf_discovery` | Source registry, ingest runs, sessions/messages, benchmark results (§5.3) | — |
| Cache/queue | No Redis. In-process exact-match LLM cache (`src/llm.py`, `temperature=0` only) and TTL cache for HF API calls; in-process background tasks | Rate-aware API cache; asynchronous benchmarks | Redis later if multi-process |
| Evaluation | LangSmith datasets/experiments, RAGAS, OpenEvals | Golden questions, groundedness and retrieval experiments | Langfuse self-hosted for sensitive traces |
| Security | Static API key on FastAPI when deployed, SQL-level category/access filters, hand-built Semantic Router guard + PII masking + NeMo Guardrails self-check (LLM calls via `src/llm.py`) | Prompt-injection testing, PII redaction, tool authorization, human approval for benchmarks | Rules-only baseline |
| Deployment | No Docker. Local run with `uv` (+ `npm` for the UI); hosted on Railway (no Dockerfile); CI via GitHub Actions | Reproducible local deployment and tests; Postgres is the external RDS instance | — |

### 3.1 LLM module (`src/llm.py`) — independent of `src`

`src/llm.py` is the **only** place that talks to an LLM. It is self-contained so it can be lifted out or tested alone:

- **Imports only** stdlib, `pyyaml` and `openai` — **never** `src.*` (not even `src.config`). Enforced by a test that greps the module's imports.
- **Configuration:** reads the `llm` block of `config/config.yaml` itself (path overridable by `LLM_CONFIG_PATH`), expands `${OPENROUTER_API_KEY}` from the environment, and fails fast if the key is missing.
- **Client:** `openai.OpenAI(base_url=llm.openrouter.base_url, api_key=...)`, created once.
- **Interface (illustrative):**

```python
def complete(messages: list[dict], *, tools: list[dict] | None = None, **kwargs) -> dict:
    """One chat completion. Returns {"content", "tool_calls", "usage", "model"}."""

def stream(messages: list[dict], **kwargs) -> Iterator[str]:
    """Token stream for SSE responses."""
```

- **Single gateway rule:** no other module imports `openai` or builds an LLM client (enforced by a test). NeMo Guardrails (`agents/nemo_llm.py`), RAGAS and openevals (`observability/llm_adapters.py`) use adapters that delegate to `src.llm.complete`.
- **Callers:** `agents/graph.py` (intent classification, synthesis, tool-calling via OpenAI-format `tools`), `agents/guards.py`, `observability/`. Callers pass messages and get plain dicts back; no OpenRouter/OpenAI types leak out.
- **Operational:** timeout from config; **3 retries** (`max_retries: 3`) with backoff on 429/5xx/timeouts, then a clear error; exact-match cache for `temperature=0` calls; log `model`, `latency_ms`, token usage and OpenRouter-reported cost into trace metadata (verify the current OpenRouter usage/cost response fields during implementation).

### 3.2 Configuration (`config/config.yaml` + `.env`)

**Non-secrets live in `config/config.yaml`; secrets live only in `.env`** (gitignored; `.env.example` lists the names). `src/config.py` (pydantic-settings) loads `.env` and merges the YAML into a single `settings` object for every `src` module except `src/llm.py`. The Slack/Groq fields in the current `src/config.py` are removed.

Target shape of `config/config.yaml` (the current file must be corrected to this — it is not valid YAML today and has `provider: "localllm"`):

```yaml
llm:
  provider: "openrouter"
  openrouter:
    base_url: "https://openrouter.ai/api/v1"
    api_key: "${OPENROUTER_API_KEY}"
    model: "google/gemini-3.5-flash-lite"
    timeout_seconds: 60
    max_retries: 3
    cache_enabled: true             # exact-match, temperature=0 only

embedding:
  model: "all-MiniLM-L6-v2"
  dim: 384

database:
  host: "localhost"
  port: 5432
  name: "tickerpulse"
  user: "postgres"
  sslmode: "require"
  schema: "hf_discovery"
  pool_size: 5

retrieval:
  mode: "dense"                     # dense | hybrid
  k: 10
  top_n: 3
  reranker: "cross-encoder/ms-marco-MiniLM-L-6-v2"   # or null
  chunker: "hierarchical"           # hierarchical | fixed50 | recursive200 | hybrid128

huggingface:
  api_timeout_seconds: 30
  max_retries: 3
  cache_ttl_seconds: 900            # 15 min; spec range 15–60 min

router:
  encoder: "all-MiniLM-L6-v2"     # Semantic Router encoder; swappable

benchmarks:
  max_runtime_seconds: 600
  max_memory_mb: 4096

api:
  auth_enabled: false               # true on Railway; key from env API_KEY
```

**Secrets in `.env`:** `OPENROUTER_API_KEY`, `TICKERPULSE_DB_PASSWORD` (existing name in `src/config.py`), `HF_TOKEN` (optional), `LANGSMITH_API_KEY`, `API_KEY` (static key, required when `api.auth_enabled`). Never commit them or return them in browser responses.

## 4. Hugging Face REST API integration requirements

**Official base:** `https://huggingface.co` (Hub API under `/api`). Prefer the maintained `huggingface_hub` SDK for pagination, authentication and metadata; REST paths below are representative and should be integration-tested against current API behavior. Implemented in `tools/hf_client.py`, exposed as tools by `tools/mcp_server.py`.

| Tool / capability | REST operation / SDK | Inputs | Output and handling |
|---|---|---|---|
| `search_models` | `GET /api/models`; `HfApi.list_models(...)` | search, task/pipeline tag, library, sort, limit | IDs, tags, downloads/likes where available; paginate, validate filters |
| `get_model_details` | `GET /api/models/{repo_id}`; `HfApi.model_info(...)` | model ID, optional revision | metadata, SHA/revision, card metadata, sibling files; never assume fields are populated |
| `search_datasets` | `GET /api/datasets`; `HfApi.list_datasets(...)` | search, task/language/license filters where supported, limit | dataset IDs and current metadata |
| `get_dataset_details` | `GET /api/datasets/{repo_id}`; `HfApi.dataset_info(...)` | dataset ID, optional revision | card metadata, configurations/file list when available |
| `get_card_section` | Hub repo file fetch (`README.md`) at pinned revision | repo type, ID, revision | current card text/section, source URL, commit SHA; no persistent copy unless permitted |
| `list_repo_files` | `HfApi.list_repo_files(...)` | repo type, ID, revision | identify README/config/sample assets without bulk downloads |
| `benchmark_embedding_model` | **Our own tool, not a Hugging Face REST endpoint** (`tools/benchmarks.py`) | model ID, approved test dataset, resource budget | Recall@k/MRR/nDCG, latency, memory, environment and model revision |

**API operating rules:** use a Hugging Face access token (`HF_TOKEN`) only where needed and permitted; keep tokens server-side; implement bounded concurrency, 3 retries with backoff for 429/5xx, configurable timeouts, pagination, and short-lived caching. Respect rate limits, robots/usage restrictions, repository gates, and per-repository licenses. Capture `fetched_at`, repository revision and source URL; mark dynamic fields such as download counts as time-sensitive. Do not assume API-provided search is semantic over full model cards.

**Suggested FastMCP tools:** `search_models`, `search_embedding_models`, `get_model_details`, `search_datasets`, `get_dataset_details`, `get_card_section`, `compare_resources`, `benchmark_embedding_model`, `search_docs` (wraps `src.retrieval.search_knowledge`), `get_section`; optionally `run_sql` for **local benchmark aggregates only** with read-only, allowlisted queries. Validate IDs, limits, tool descriptions, and result sizes. Start with local stdio; remote deployment requires authenticated transport, network restrictions and audit logs.

## 5. One-time public knowledge collection and incremental refresh

### 5.1 Seed corpus (one-time bootstrap) — `src/ingest.py`

1. **Source registry** (`config/sources.yaml`, mirrored into `hf_discovery.sources`): Select ~100–200 pages of *official, publicly accessible, license-checked* documentation from **Transformers, Datasets, Sentence Transformers and the Hugging Face Hub**, taken from each project's GitHub docs source at the **latest release tag at bootstrap** (commit SHA recorded). Optionally include a small curated set of model/dataset cards **only if their licensing/terms permit indexing**. Record each in `hf_discovery.sources`: `source_id`, canonical URL, owner, license, access class, permitted-use decision, version and attribution.
2. **Fetch at pinned version:** Retrieve documentation from an official repository/release or supported documentation URL; record commit/revision, content hash, retrieval timestamp and HTTP status. Reject gated/private content and unknown-license content from persistent indexing until reviewed.
3. **Parse and normalize:** Markdown/HTML extraction; Docling for PDFs; repair heading hierarchy, preserve code snippets and heading breadcrumbs. For scans, use regex/text extraction → OCR → VLM only as needed; keep provenance.
4. **Chunk (`src/chunking.py`):** Default section/hierarchy-aware chunks with `heading_path` and source offsets. A/B test fixed 50-word, RecursiveCharacterTextSplitter (200/20), HybridChunker (128 tokens) and hierarchical chunking. Ensure headings remain attached to chunks.
5. **Embed and insert:** Embed approved chunks with `src.embed.embed()`; deterministic `chunk_id` = sha256(`source_id | revision | chunk_strategy | chunk_index | content_hash`). Insert into `hf_discovery.doc_chunks` under a new `ingest_runs` row (`status='running'`); `INSERT ... ON CONFLICT (chunk_id) DO NOTHING` makes re-runs idempotent. Store chunk text only when `sources.store_text` allows it.
6. **Validate and publish:** Check chunk counts, duplicate hashes, orphan URLs, citation resolvability and retrieval smoke tests. Publish by flipping the run to `status='active'` (previous active run → `superseded`) in one transaction — at most one run is `active` (enforced by a unique index). Rollback = re-activate the prior run. **Experiments** (e.g. the 16-variant study) keep several runs, one per chunker, queryable at once: retrieval accepts an optional `run_id` that overrides the `active` default.

Store the chunk text only when the documented permission decision allows it; otherwise retain source references and retrieve the page live at question time. **Do not put current downloads, likes, entire datasets or model weights in the database.**

### 5.2 Refresh strategy

- **Bootstrap once** for the selected versioned documentation corpus; avoid crawling the entire Hub.
- **Periodic metadata check** (e.g., weekly for docs): compare ETag/commit/content hash; re-embed only changed chunks, delete removed chunks, and set `deleted_at` tombstones for withdrawn/restricted content.
- **On-demand live fetch** for model/dataset discovery and up-to-date model cards; short TTL cache (e.g., 15–60 minutes, configurable via `huggingface.cache_ttl_seconds`) for frequently requested public metadata.
- **Scheduled index audit** for broken citations, license changes, version drift and duplicate vectors. `hf_discovery.sources` is the authority for index eligibility; Hub remains authority for live resource metadata.

### 5.3 Database schema — `src/schema.sql` (new schema `hf_discovery`)

New schema on the existing RDS instance. **No existing schema or table (e.g. `slackbot.*`) is read or modified.** `src/db.py` keeps its pool and `--init` / status CLI, with `search_path` changed to `hf_discovery,public` and the status query pointed at `hf_discovery`. The schema name comes from `database.schema` in `config.yaml`.

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS hf_discovery;

-- License/permission registry: authority for what may be indexed.
CREATE TABLE IF NOT EXISTS hf_discovery.sources (
    source_id      text PRIMARY KEY,
    canonical_url  text NOT NULL,
    owner          text,
    license        text,
    access_class   text NOT NULL DEFAULT 'public',
    permitted_use  text NOT NULL CHECK (permitted_use IN
                   ('pending_review','index_text','reference_only','rejected')),
    store_text     boolean NOT NULL DEFAULT false,
    version        text,
    attribution    text,
    category       text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now()
);

-- One row per ingest; exactly one 'active' run is what retrieval reads.
CREATE TABLE IF NOT EXISTS hf_discovery.ingest_runs (
    run_id                    bigserial PRIMARY KEY,
    corpus_revision           text NOT NULL UNIQUE,
    chunk_strategy            text NOT NULL,
    embedding_model           text NOT NULL,
    embedding_model_revision  text,
    status                    text NOT NULL CHECK (status IN
                              ('running','validated','active','superseded','failed')),
    stats                     jsonb NOT NULL DEFAULT '{}',
    started_at                timestamptz NOT NULL DEFAULT now(),
    finished_at               timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS ingest_runs_one_active
    ON hf_discovery.ingest_runs ((true)) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS hf_discovery.doc_chunks (
    chunk_id        text PRIMARY KEY,          -- sha256(source_id|revision|chunk_strategy|chunk_index|content_hash)
    run_id          bigint NOT NULL REFERENCES hf_discovery.ingest_runs,
    source_id       text   NOT NULL REFERENCES hf_discovery.sources,
    source_url      text   NOT NULL,
    heading_path    text[] NOT NULL DEFAULT '{}',
    revision        text   NOT NULL,
    license         text   NOT NULL,
    category        text,
    access          text   NOT NULL DEFAULT 'public',
    chunk_strategy  text   NOT NULL,
    chunk_index     int    NOT NULL,
    start_offset    int,
    end_offset      int,
    content         text,                      -- NULL when sources.store_text = false
    content_hash    text   NOT NULL,
    embedding       vector(384) NOT NULL,      -- all-MiniLM-L6-v2
    tsv             tsvector GENERATED ALWAYS AS
                    (to_tsvector('english', coalesce(content, ''))) STORED,
    indexed_at      timestamptz NOT NULL DEFAULT now(),
    deleted_at      timestamptz                -- tombstone
);
CREATE INDEX IF NOT EXISTS doc_chunks_embedding_hnsw
    ON hf_discovery.doc_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS doc_chunks_tsv_gin ON hf_discovery.doc_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS doc_chunks_run     ON hf_discovery.doc_chunks (run_id);
CREATE INDEX IF NOT EXISTS doc_chunks_source  ON hf_discovery.doc_chunks (source_id);

CREATE TABLE IF NOT EXISTS hf_discovery.sessions (
    thread_id       uuid PRIMARY KEY,          -- pseudonymous
    memory_enabled  boolean NOT NULL DEFAULT true,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS hf_discovery.messages (
    message_id        bigserial PRIMARY KEY,
    thread_id         uuid NOT NULL REFERENCES hf_discovery.sessions ON DELETE CASCADE,
    role              text NOT NULL CHECK (role IN ('user','assistant','tool')),
    content           text NOT NULL,
    pipeline_variant  text,
    citations         jsonb NOT NULL DEFAULT '[]',
    guard_events      jsonb NOT NULL DEFAULT '[]',
    llm_model         text,
    token_usage       jsonb,
    cost_usd          numeric(12,6),
    latency_ms        int,
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_thread ON hf_discovery.messages (thread_id, created_at);

CREATE TABLE IF NOT EXISTS hf_discovery.benchmark_runs (
    job_id            uuid PRIMARY KEY,
    model_id          text NOT NULL,
    model_revision    text,
    dataset_id        text NOT NULL,
    dataset_revision  text,
    status            text NOT NULL CHECK (status IN ('queued','running','succeeded','failed','timeout')),
    seed              int,
    hardware          jsonb,
    metrics           jsonb,                   -- Recall@k, MRR, nDCG, latency, memory
    error             text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    finished_at       timestamptz
);
```

Notes: HNSW requires pgvector ≥ 0.5 — `python -m src.db` prints the installed version; fall back to IVFFlat if older. `tsv` covers stored `content` only, so `reference_only` sources are dense-searchable but not keyword-searchable.

## 6. RAG pipeline and swappable experiments

```text
User question → classify → query rewrite (optional) → category/access filters (SQL WHERE)
  → dense (pgvector cosine; tuned threshold) OR dense + Postgres FTS (RRF fusion)
  → retrieve k=10 → cross-encoder rerank → top_n=3
  → source verification + sensitive-content redaction
  → prompt with citations → src/llm.py (OpenRouter) → citation/guard checks
```

`src/retrieval.py::search_knowledge(query, category=None, k=10, mode="dense", run_id=None)` evolves from the old `src/tools.py` function (renamed, since `tools/` now holds the MCP server): it embeds with `src.embed.embed`, queries `hf_discovery.doc_chunks` joined to the `ingest_runs` row given by `run_id` (default: the `active` run) with `deleted_at IS NULL`, and returns `chunk_id, source_url, heading_path, revision, content, score`. Hybrid mode (illustrative):

```sql
WITH dense AS (
  SELECT chunk_id, row_number() OVER (ORDER BY embedding <=> %(q_vec)s) AS r
  FROM active_chunks ORDER BY embedding <=> %(q_vec)s LIMIT %(k)s),
lex AS (
  SELECT chunk_id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
  FROM active_chunks, websearch_to_tsquery('english', %(q_text)s) q
  WHERE tsv @@ q ORDER BY ts_rank_cd(tsv, q) DESC LIMIT %(k)s)
SELECT chunk_id, sum(1.0 / (60 + r)) AS rrf
FROM (SELECT * FROM dense UNION ALL SELECT * FROM lex) u
GROUP BY chunk_id ORDER BY rrf DESC LIMIT %(k)s;
```

| Stage | Default | A/B alternative | Success measurement |
|---|---|---|---|
| Parse | Markdown parser / Docling PDFs, heading repair | Plain extraction; OCR/VLM fallback on scans | Heading accuracy, text coverage, cost/page |
| Chunk | Hierarchical + breadcrumb | Fixed 50 words; recursive 200/20; Hybrid 128 tokens | Recall@5, context precision, citation granularity |
| Retrieve | Dense MiniLM + pgvector, SQL filters | Hybrid dense + Postgres FTS (RRF); tuned thresholds | Recall@k, MRR, p95 latency |
| Rerank | Cross-encoder k=10 → 3 (`src/rerank.py`, model from config) | None | nDCG@3, answer faithfulness, latency |
| Structured | Direct Hugging Face API / local SQL for aggregates | LLM-only baseline (eval only) | Exact filter/count correctness |
| Multimodal | Out of MVP; text from image/PDF extraction | CLIP for actual image-query benchmark | Image Recall@k |
| LLM | OpenRouter `google/gemini-3.5-flash-lite` via `src/llm.py` | Alternate OpenRouter model (config change only) | Answer relevancy, groundedness, cost, p95 latency |

All variants are versioned configuration switches in `config.yaml` (`parse_mode`, `retrieval.chunker`, `retrieval.mode`, `retrieval.reranker`, `llm.openrouter.model`, `cache_mode`) so experiments can reproduce results. Each chunker is a separate `ingest_runs` row. Use one fixed golden dataset and pinned corpus/model revisions. Full text-only factorial baseline: **4 chunkers × 2 retrieval modes × 2 rerank modes = 16 variants**; parsing and model routing are separate controlled experiments.

## 7. Agent workflows, backend contracts and UI

**LangGraph nodes (`agents/graph.py`):** a `guard_input` node (Semantic Router guard, PII masking, NeMo self-check) runs first, then `classify_intent` (Semantic Router, encoder from `router.encoder`) → `retrieve_docs` and/or `call_mcp_tools` → `validate_evidence` → `synthesize` → `verify_citations` → `respond`. For benchmark requests, add `authorize_benchmark` (LangGraph `interrupt()` human approval — nothing runs without it) → `enqueue_job` → `poll_results`. Memory uses the Postgres checkpointer keyed by `thread_id`. LLM-backed nodes call `src.llm.complete` / `src.llm.stream`. Keep tool outputs as untrusted data, not instructions.

**Illustrative API contracts (`src/api.py`; when `api.auth_enabled`, every `/api/*` call needs header `X-API-Key`, compared with `secrets.compare_digest` to env `API_KEY`):**

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | `{message, thread_id, pipeline_variant}` → streamed answer, cited chunks, guard events |
| `GET /api/models/search` | UI filterable model discovery via MCP |
| `GET /api/datasets/search` | Dataset discovery via MCP |
| `POST /api/compare` | Compare model IDs with cited specs and missing-field warnings |
| `POST /api/benchmarks` | Start allowlisted asynchronous evaluation; return job ID (`benchmark_runs`) |
| `GET /api/benchmarks/{job_id}` | Results, metrics, versions, hardware and provenance |
| `GET /api/evals/variants` | RAG variant leaderboard |

**UI panels:** chat + clickable citations and heading breadcrumbs; discovery filters (task, license, language, resource budget); side-by-side model comparison; benchmark metrics with **measured vs published** labels; expandable retrieved chunks/rerank scores; pipeline-variant selector; guardrail-event badge; thread memory on/off toggle. Use a checkpointer keyed by `thread_id`; do not treat conversational memory as a trusted source.

## 8. Security, evaluation and observability

| Capability | Default | Alternative / A/B | Metric and attack/test |
|---|---|---|---|
| Input routing | Hand-built Semantic Router guard node + schema checks | LLM intent classification | Intent F1; adversarial ambiguous queries |
| Input guard | NeMo Guardrails self-check input | Rules-only baseline | Unsafe-request block rate; false positives |
| Injection defense | Hardened system prompt, retrieved-text delimiters, output regex checks | Add LLM judge/policy classifier | Direct/indirect injection attack-success rate; injected model-card chunk test |
| Privacy | PII masking, retrieval-time redaction, SQL category/access filters | Redaction at ingestion | PII leakage rate; unauthorized chunk exposure = 0 |
| Tool safety | Read-only default; allowlisted tools; human approval for risky execution; parameterized SQL only | Tool-call policy variants | Unauthorized action success = 0; approval bypass tests |
| Output handling | `html.escape` for untrusted text where rendered as HTML; UI safe rendering | Sanitizer library | XSS test pass rate |
| Observability | LangSmith `@traceable` around parse/retrieve/rerank/tool/generate/guard | Self-hosted Langfuse | Trace completeness, p95 stage latency, errors, token cost |

Guards live in `agents/guards.py`; PII masking, the Semantic Router guard and human approval are hand-built graph nodes/functions (the course's LangChain middleware is not used with the raw SDK).

**Evaluation:** Create golden Q&A cases in LangSmith (`client.create_dataset`), run reproducible experiments (`client.evaluate`), and report RAGAS faithfulness, answer relevancy, context precision/recall; custom AspectCritic/RubricsScore; heuristic citation correctness; OpenEvals LLM-as-judge (judge calls go through `src/llm.py`). Include model-discovery tests for task/license/size constraints and benchmark reproducibility. Produce a 16-row chunking × retrieval × reranking comparison with quality, latency, and cost; select the best *measured* configuration rather than presupposing the default wins.

**Trace metadata:** `request_id`, `thread_id` (pseudonymous), `pipeline_variant`, `prompt_version`, `embedding_model_revision`, `corpus_revision`, `hf_api_status`, `tool_name`, `latency_ms`, `token_usage`, `estimated_cost_usd` (from OpenRouter usage), `guardrail_events`. Redact secrets and PII from logs; use Langfuse if self-hosting is required for sensitive telemetry.

## 9. Deployment, repository layout and configuration

```text
repo/
  config/config.yaml        # non-secret settings (§3.2); config/sources.yaml = seed-corpus registry
  .env                      # secrets only, gitignored; .env.example committed
  src/                      # core libraries
    config.py               # pydantic-settings: .env secrets + config.yaml → `settings`
    db.py                   # psycopg_pool ConnectionPool; `python -m src.db --init`
    schema.sql              # hf_discovery DDL (§5.3)
    embed.py                # all-MiniLM-L6-v2 (from config), 384-dim, loaded once
    llm.py                  # OpenRouter client; imports NO src.* modules (§3.1)
    chunking.py             # 4 chunker strategies
    ingest.py               # registry → fetch → parse → chunk → embed → insert → publish
    retrieval.py            # search_knowledge (dense/hybrid, optional run_id)
    rerank.py               # cross-encoder rerank
    prompts.py              # system/answer prompts, delimiters
    api.py                  # FastAPI routes, schemas, SSE streaming, API-key auth
    auth.py                 # static API-key dependency
    main.py                 # entrypoint (uvicorn)
  agents/                   # LangGraph state, router, nodes, guards, NeMo LLM adapter
  tools/                    # hf_client.py, mcp_server.py (FastMCP), benchmarks.py
  observability/            # tracing, evals, llm_adapters.py, golden/, reports/
  ui/                       # React chat/discovery/compare UI (Vite + TypeScript)
  standalone-scripts/       # one-off CLIs: ingest, ask, run_evals, benchmark dataset fetch
  tests/                    # unit, contract, integration, security
  specs/  tasks/
```

**Module dependency rules:** `src.config` ← `src.db`, `src.embed` ← `src.retrieval`, `src.ingest`, `src.rerank` ← `agents` ← `src.api`/`src.main`; `tools.mcp_server` → `tools.hf_client`, `src.retrieval`, `tools.benchmarks`. `src.llm` is a leaf with **no** `src` imports and is used by `agents` and `observability`. No circular imports; each module is runnable or testable alone (`python -m src.<module>` self-checks where useful).

**Changes to existing files:**
- `src/config.py` — remove Slack/Groq fields; add YAML loading; keep `tickerpulse_db_password` and DSN building; read DB host/schema from `config.yaml`.
- `src/db.py` — `search_path` and status query move from `slackbot` to `hf_discovery`.
- `src/embed.py` — read model name from `settings.embed_model` instead of the hard-coded string.
- `src/llm.py` — rewrite as §3.1 (current file is an incomplete fragment).
- `src/tools.py` — renamed to `src/retrieval.py`; keep and evolve `search_knowledge`; **remove** `get_table_schema` and `validate_sql` (left over from another project).
- `config/config.yaml` — rewrite to the §3.2 shape.

**Deployment topology:** No Docker. Locally: `uv run` for API/agent (the MCP server starts as a stdio subprocess) and `npm run dev` for the UI. Hosted on Railway (no Dockerfile) with `api.auth_enabled: true` and secrets as Railway env vars. PostgreSQL is the existing external RDS instance (schema `hf_discovery`). CI (GitHub Actions) runs lint, tool-contract tests, the `src/llm.py` isolation and single-gateway checks, retrieval smoke tests and prompt-injection tests. Model benchmarks run in a subprocess with wall-clock and memory limits from config, allowlisted models only, no `trust_remote_code`.

## 10. Milestones and acceptance criteria

| Milestone | Deliverable | Exit criterion |
|---|---|---|
| M1 — Foundations | `config.yaml`/`config.py` cleanup, `hf_discovery` schema applied, `src/llm.py` OpenRouter call, React/FastAPI/LangGraph shell, FastMCP HF search tools | `python -m src.db --init` creates all `hf_discovery` tables; one OpenRouter completion succeeds; search models/datasets and render provenance end-to-end |
| M2 — Public-doc RAG | Source registry, one-time ingestion, pgvector + FTS, cited Q&A | ≥100 approved docs/pages indexed in an `active` run; citations resolve to correct source sections |
| M3 — Discovery & comparison | Live API details, filters, model comparisons | Constraint tests pass; dynamic fields show fetched timestamp |
| M4 — Evaluation & security | 16-variant leaderboard, guardrails, red-team tests | All variants run reproducibly; zero unauthorized tool execution in test set |
| M5 — Benchmarks & release | Isolated embedding benchmark, UI results, observability, Railway deploy | Re-run benchmark within stated tolerance; end-to-end demo runs from README; Railway deploy rejects unauthenticated requests |

**Proposed measurable MVP targets (validate with baseline):** citation correctness ≥95% on 50 curated questions; Recall@5 ≥85% on answerable document questions; groundedness/faithfulness ≥0.90 using a fixed evaluator; p95 cached-document Q&A <8 seconds under demo load; 100% of displayed comparisons label source and retrieval time; 100% of red-team tool-escalation cases blocked; benchmark runs capture model revision, dataset revision, hardware, seed, latency and retrieval metrics. Report actual scores, confidence limits and failure cases—do not claim targets are achieved until measured.

## 11. Decisions log and open questions

**Confirmed with developer:**
- Core libraries in `src/`; `agents/`, `tools/`, `observability/`, `ui/`, `standalone-scripts/` at repo root. React/FastAPI/LangGraph/FastMCP stack retained.
- OpenRouter only, via `openai` SDK; LiteLLM dropped; single model `google/gemini-3.5-flash-lite`; `src/llm.py` imports no other `src` module and is the **only** LLM entry point (libraries like NeMo/RAGAS/openevals go through adapters).
- Course LangChain middleware not used; PII masking, human approval and guardrails are hand-built and required.
- Everything tunable in `config.yaml`; secrets only in `.env`. 3 retries before erroring out; other limits use the defaults in §3.2.
- PostgreSQL + pgvector replaces Qdrant; hybrid = pgvector + Postgres full-text (no rank-bm25). Postgres checkpointer for thread memory. No Redis.
- Same RDS instance, **new schema `hf_discovery`**, new tables (sources, ingest_runs, doc_chunks, sessions, messages, benchmark_runs) with typed provenance columns; existing tables not reused.
- Publishing via `ingest_runs.status='active'`, with optional `run_id` override for experiments.
- Embeddings and Semantic Router encoder: `all-MiniLM-L6-v2` (384-dim), both configurable.
- `get_table_schema` / `validate_sql` dropped; `src/tools.py` → `src/retrieval.py`.
- No Docker; Railway hosting; static API key auth when deployed, none locally.
- Seed corpus: Transformers, Datasets, Sentence Transformers, Hub docs at latest release tags. Benchmark dataset comes from a developer-provided standalone script.

**Open — verify during implementation (T3):**
- Installed pgvector version on RDS (HNSW vs IVFFlat).
- Whether `CREATE EXTENSION vector` / `CREATE SCHEMA` can run as `postgres` on RDS.

## 12. Primary official references

- Hugging Face Hub API: https://huggingface.co/docs/hub/api
- Hugging Face Hub Python client: https://huggingface.co/docs/huggingface_hub/index
- Hugging Face model cards: https://huggingface.co/docs/hub/model-cards
- Hugging Face dataset cards: https://huggingface.co/docs/hub/datasets-cards
- Hugging Face terms: https://huggingface.co/terms-of-service
- Model Context Protocol: https://modelcontextprotocol.io/
- LangGraph: https://docs.langchain.com/oss/python/langgraph/overview
- pgvector: https://github.com/pgvector/pgvector
- PostgreSQL full-text search: https://www.postgresql.org/docs/current/textsearch.html
- OpenRouter API: https://openrouter.ai/docs
- LangSmith: https://docs.langchain.com/langsmith/

*References are implementation pointers. Verify current SDK signatures, API limits, licenses, and deployment policies during development.*
