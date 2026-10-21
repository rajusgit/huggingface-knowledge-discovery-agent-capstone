# Task List — HF Knowledge & Discovery Agent

Plan and decisions: `tasks/plan.md`. Course references are relative to `../ai-engineering-bootcamp-codebasics/`.
Commands assume `uv` from repo root. Unit tests: `uv run pytest tests/<file> -q`. UI build: `cd ui && npm run build`.

---

## Phase 0: Foundation

### - [x] T1: Align architecture spec with decisions D1–D10
**Description:** Update the technical architecture spec so it matches the confirmed decisions: folder layout (D1), raw-SDK LLM with no middleware (D2/D3), single LLM gateway (D2a), everything configurable (D3a), 3 retries (D3b), no Redis/Docker (D8/D9), Railway + auth switch (D9a), Postgres checkpointer (D7), seed corpus (D11), retrieval `run_id` override for experiments.
**Acceptance criteria:**
- [ ] §9 layout shows `src/`, `agents/`, `tools/`, `observability/`, `ui/`, `standalone-scripts/`
- [ ] No remaining references to Redis, Docker, LiteLLM, Qdrant or LangChain middleware as defaults
- [ ] Checkpointer, `run_id` override, single-gateway rule, config defaults table and Railway auth are documented
**Verification:** `grep -nE "Redis|Docker|LiteLLM|Qdrant|Middleware" specs/*architecture.md` shows only "replaced/removed" notes
**Dependencies:** None · **Files:** `specs/HuggingFace_Knowledge_Discovery Agent_technical_architecture.md` · **Scope:** XS

### - [x] T2: Project skeleton + config
**Description:** `pyproject.toml` (uv, Python ≥3.11) with root packages `src`, `agents`, `tools`, `observability`; rewrite `config/config.yaml` to the spec §3.2 shape **plus every tunable (D3a) with the defaults in plan.md** (`max_retries: 3`, timeouts, TTLs, benchmark limits, `router.encoder`, reranker, k/top_n, chunk sizes, `api.auth_enabled: false`); rewrite `src/config.py` (YAML + `.env` merge, drop Slack/Groq fields); add `.env.example`, `.gitignore`.
**Acceptance criteria:**
- [ ] `from src.config import settings` exposes `db`, `embedding`, `retrieval`, `router`, `huggingface`, `benchmarks`, `api` sections and the DSN
- [ ] Missing `TICKERPULSE_DB_PASSWORD` fails with a clear error; `.env` is gitignored
- [ ] `config.yaml` parses (`yaml.safe_load`), has `provider: openrouter` and `max_retries: 3`
- [ ] Guard test: no model IDs or tunables hard-coded in `src/`, `agents/`, `tools/`, `observability/` (grep for known model names / numeric literals in retrieval and HTTP calls)
**Verification:** `uv sync && uv run python -c "from src.config import settings; print(settings.db.schema)"` prints `hf_discovery`; `uv run pytest tests/test_config.py tests/test_no_hardcoding.py -q`
**Dependencies:** None · **Files:** `pyproject.toml`, `config/config.yaml`, `src/config.py`, `.env.example`, `tests/test_config.py`, `tests/test_no_hardcoding.py` · **Scope:** S

### - [ ] T3 (code + offline tests done, **awaiting sign-off to run `--init` on RDS**): `hf_discovery` schema + db init
**Description:** Add `src/schema.sql` (spec §5.3; the LangGraph Postgres checkpointer tables via `PostgresSaver.setup()` are created in T10, when that dependency is added); point `src/db.py` at `hf_discovery`. High-risk first: proves RDS permissions and pgvector version.
**Acceptance criteria:**
- [ ] `python -m src.db --init` creates all 6 tables + indexes idempotently (second run is a no-op)
- [ ] Status output prints the pgvector version and `hf_discovery.*` tables only; `slackbot.*` untouched
- [ ] HNSW index created, or IVFFlat fallback if pgvector < 0.5
**Verification:** `uv run python -m src.db --init` twice; manual: `\dt hf_discovery.*` in psql
**Dependencies:** T2 · **Files:** `src/schema.sql`, `src/db.py` · **Scope:** S

### - [ ] T4: `src/llm.py` OpenRouter client
**Description:** `complete(messages, tools=None, **kw) -> dict` and `stream(...)` using `openai.OpenAI(base_url=OpenRouter)`; reads the `llm` block of config.yaml itself (model, timeout, retries, cache switch); **3 retries with backoff** on 429/5xx/timeouts, then raises a clear error; exact-match in-process cache for `temperature=0` calls (course: 16_advanced_ai_eng). Returns `{content, tool_calls, usage, model, latency_ms}`. This is the **only** LLM entry point in the project (D2a).
**Acceptance criteria:**
- [ ] Module imports only stdlib, `yaml`, `openai` (test parses its AST and fails on any `src.` import)
- [ ] Gateway test: no other module in the repo imports `openai` or constructs an LLM client
- [ ] A mocked 3×-failing call makes exactly 4 attempts (1 + 3 retries) then raises; missing `OPENROUTER_API_KEY` raises before any network call
- [ ] Second identical `temperature=0` call is served from cache (no network)
**Verification:** `uv run pytest tests/test_llm.py -q` (mocked client); manual: `uv run python -m src.llm "say hi"` returns text + usage
**Dependencies:** T2 (config.yaml only) · **Files:** `src/llm.py`, `tests/test_llm.py` · **Scope:** S

### Checkpoint A — Foundation
- [ ] `uv run pytest -q` green
- [ ] Schema exists on RDS; one live OpenRouter completion succeeded
- [ ] Config defaults from plan.md present in `config.yaml`; no-hardcoding and LLM-gateway tests green
- [ ] Review with developer before Phase 1

---

## Phase 1: Thin Learn slice

### - [ ] T5: Ingest one approved doc → active run
**Description:** Minimal `src/ingest.py` + `src/chunking.py` (hierarchical only): register one source in `sources`, fetch at a pinned revision, parse with Docling, `HierarchicalChunker` with breadcrumb (course: evals_with_ragas `convert_chunk`), embed with `src/embed.py`, insert under a new `ingest_runs` row, publish as `active`. CLI in `standalone-scripts/ingest.py`.
**Acceptance criteria:**
- [ ] Chunks carry `source_url`, `heading_path`, `revision`, `license`, `content_hash`, deterministic `chunk_id`
- [ ] Re-running inserts 0 new rows (idempotent); `reference_only` sources store `content = NULL`
- [ ] Sources with `permitted_use` other than `index_text`/`reference_only` are refused
**Verification:** `uv run python standalone-scripts/ingest.py --source <id>` then row count > 0; `uv run pytest tests/test_chunking.py -q`
**Dependencies:** T3 · **Files:** `src/ingest.py`, `src/chunking.py`, `src/embed.py` (read model from config), `standalone-scripts/ingest.py`, `tests/test_chunking.py` · **Scope:** M

### - [ ] T6: Dense retrieval + cited answer CLI
**Description:** Evolve `search_knowledge` into `src/retrieval.py` (Q1, agreed; `src/tools.py` is deleted): pgvector cosine over the active run (or a given `run_id`), `score_threshold`, category filter. Answer prompt with numbered context and required citations (course: rag.ipynb `retrieve → build_context`). Remove `get_table_schema`/`validate_sql`.
**Acceptance criteria:**
- [ ] Returns `chunk_id, source_url, heading_path, revision, content, score`, excluding `deleted_at` rows
- [ ] Answer cites `[n]` markers that map to returned chunks; with no chunk above threshold it says it can't answer
**Verification:** `uv run python standalone-scripts/ask.py "How do I set max_new_tokens?"` shows answer + URL + breadcrumb; `uv run pytest tests/test_retrieval.py -q`
**Dependencies:** T4, T5 · **Files:** `src/retrieval.py` (replaces `src/tools.py`), `src/prompts.py`, `standalone-scripts/ask.py`, `tests/test_retrieval.py` · **Scope:** M

### - [ ] T7: FastAPI `/api/chat` (RAG-only, SSE)
**Description:** `src/api.py` with `POST /api/chat {message, thread_id, pipeline_variant}` → SSE stream of tokens, then a final event with citations. CORS for the Vite dev origin (course: mcp-client `fastapi_server.py`). `src/main.py` runs uvicorn.
**Acceptance criteria:**
- [ ] Streams tokens; final event JSON has `citations[]` and `guard_events[]` (empty for now)
- [ ] Pydantic validation rejects empty/oversized messages with 422
**Verification:** `uv run python -m src.main` + `curl -N -X POST localhost:8000/api/chat -d '{...}'`; `uv run pytest tests/test_api.py -q` (TestClient)
**Dependencies:** T6 · **Files:** `src/api.py`, `src/main.py`, `tests/test_api.py` · **Scope:** S

### - [ ] T8: React chat shell + citation drawer
**Description:** Vite + React + TS + Tailwind in `ui/`: chat panel consuming the SSE stream, citation drawer with clickable URL + heading breadcrumb + score. Untrusted text rendered as text (no `dangerouslySetInnerHTML`).
**Acceptance criteria:**
- [ ] Streamed answer renders incrementally; citations open the source URL in a new tab
- [ ] API base URL comes from `VITE_API_URL`
**Verification:** `cd ui && npm run build`; manual: ask a question in the browser and click a citation
**Dependencies:** T7 · **Files:** `ui/package.json`, `ui/src/App.tsx`, `ui/src/api.ts`, `ui/src/components/Chat.tsx`, `ui/src/components/Citations.tsx` · **Scope:** M

### Checkpoint B — Learn path
- [ ] Browser question → streamed, cited answer from the one ingested doc
- [ ] All tests pass; UI builds
- [ ] Review with developer

---

## Phase 2: Discover

### - [ ] T9: HF client + FastMCP server
**Description:** `tools/hf_client.py` wraps `huggingface_hub.HfApi` (`list_models`, `model_info`, `list_datasets`, `dataset_info`) with configurable timeout, **3 retries with backoff** on 429/5xx then a clear error, configurable TTL cache, input validation; adds `fetched_at` + `source_url` to every result. `tools/mcp_server.py` exposes `search_models`, `search_embedding_models`, `get_model_details`, `search_datasets`, `get_dataset_details`, `search_docs` over stdio (course: hr-assist `server.py`).
**Acceptance criteria:**
- [ ] Every result has `fetched_at`, `revision` (when available), `source_url`; missing fields stay null and are never invented
- [ ] Invalid repo IDs / limits > max are rejected before any API call
- [ ] Server lists all tools in MCP Inspector
**Verification:** `uv run pytest tests/test_hf_client.py -q` (mocked); manual: `npx @modelcontextprotocol/inspector uv run python -m tools.mcp_server`
**Dependencies:** T2 (and T6 for `search_docs`) · **Files:** `tools/__init__.py`, `tools/hf_client.py`, `tools/mcp_server.py`, `tests/test_hf_client.py` · **Scope:** M

### - [ ] T10: LangGraph agent
**Description:** `agents/graph.py`: `classify_intent` (Semantic Router, course Session 6/9; encoder from `router.encoder` in config, MiniLM for now; route thresholds also configurable) → `retrieve_docs` | `call_mcp_tools` | both → `synthesize` (via `src.llm.complete`, tool calls in OpenAI format; MCP tools loaded with `MultiServerMCPClient` and converted with `convert_to_openai_tool`) → `verify_citations` → `respond`. Postgres checkpointer keyed by `thread_id` (D7). Tool outputs are wrapped as untrusted data.
**Acceptance criteria:**
- [ ] Doc questions route to RAG, model/dataset searches to MCP, mixed questions to both, with separate source attribution
- [ ] Follow-up ("how does the second one compare?") resolves using thread memory; memory off → no history loaded
- [ ] Routing accuracy measured on a 30-query labelled set (target ≥95%)
**Verification:** `uv run pytest tests/test_agent_routing.py -q`; manual: `uv run python standalone-scripts/agent_cli.py`
**Dependencies:** T4, T6, T9 · **Files:** `agents/__init__.py`, `agents/graph.py`, `agents/router.py`, `agents/state.py`, `tests/test_agent_routing.py` · **Scope:** M

### - [ ] T11: Discovery endpoints + agent-backed chat
**Description:** `/api/chat` now invokes the agent (streaming per node, course: react_langgraph `stream_mode`); add `GET /api/models/search` and `GET /api/datasets/search` (direct to `tools.hf_client` for UI filters). Persist messages to `hf_discovery.messages`.
**Acceptance criteria:**
- [ ] Chat SSE events include node progress, citations and tool results with `fetched_at`
- [ ] Search endpoints accept task/library/license/language/sort/limit filters and validate them
**Verification:** `uv run pytest tests/test_api.py -q`; manual curl of both search endpoints
**Dependencies:** T7, T10 · **Files:** `src/api.py`, `tests/test_api.py` · **Scope:** S

### - [ ] T12: UI discovery panel + memory toggle
**Description:** Discovery page with filters (task, license, language, size budget), model/dataset result cards with `fetched_at` and a "time-sensitive" label on downloads/likes; memory on/off toggle (default off) that sends a fresh `thread_id` when off.
**Acceptance criteria:**
- [ ] Filters map 1:1 to API query params; empty results show a no-results state
- [ ] Memory toggle off → each message uses a new `thread_id`
**Verification:** `cd ui && npm run build`; manual: search "multilingual sentence-similarity" and toggle memory
**Dependencies:** T8, T11 · **Files:** `ui/src/components/Discover.tsx`, `ui/src/components/ResultCard.tsx`, `ui/src/App.tsx`, `ui/src/api.ts` · **Scope:** M

### Checkpoint C — Discover path
- [ ] Live HF results in UI with `fetched_at`; agent routes correctly; follow-ups use memory
- [ ] All tests pass; review with developer

---

## Phase 3: Compare + Evaluate

### - [ ] T13: Compare tools + endpoint + UI table
**Description:** `get_card_section` (README at pinned revision, live fetch, not persisted) and `compare_resources` (side-by-side facts: task, license, params/size, languages, revision, plus missing-field warnings) in the MCP server; `POST /api/compare`; UI compare table that labels every cell with its source and `fetched_at`.
**Acceptance criteria:**
- [ ] Missing card/metadata fields appear as explicit "not stated" warnings
- [ ] User constraints (license, CPU-friendly size, language) are enforced and violations flagged
**Verification:** `uv run pytest tests/test_compare.py -q`; manual: compare two embedding models in UI
**Dependencies:** T9, T11 · **Files:** `tools/hf_client.py`, `tools/mcp_server.py`, `src/api.py`, `ui/src/components/Compare.tsx`, `tests/test_compare.py` · **Scope:** M

### - [ ] T14: Benchmark runner + `benchmark_runs`
**Description:** `tools/benchmarks.py`: runs an allowlisted sentence-transformers model on an approved small retrieval dataset in a **subprocess** (no Docker) with wall-clock timeout + `resource` memory limit from config (`benchmarks.max_runtime_seconds`, `max_memory_mb`), no `trust_remote_code`. The real dataset comes from the developer's standalone fetch script (D12, pending); build and test against a tiny in-repo fixture until it arrives. Records Recall@k, MRR, nDCG, latency, memory, seed, hardware, model/dataset revisions in `benchmark_runs`. Exposed as MCP tool `benchmark_embedding_model`.
**Acceptance criteria:**
- [ ] Same model + dataset + seed re-run reproduces metrics within a stated tolerance
- [ ] Non-allowlisted model or a timeout → `status` `failed`/`timeout` with error, no crash
**Verification:** `uv run pytest tests/test_benchmarks.py -q` (tiny fixture); manual: `uv run python -m tools.benchmarks --model sentence-transformers/all-MiniLM-L6-v2`
**Dependencies:** T3, T9; real dataset blocked on developer script (D12) · **Files:** `tools/benchmarks.py`, `tools/mcp_server.py`, `tests/test_benchmarks.py`, `tests/fixtures/bench_tiny.json` · **Scope:** M

### - [ ] T15: Benchmark approval (human-in-the-loop, required) + API + UI
**Description:** Human approval is a required deliverable (D3), hand-built because `HumanInTheLoopMiddleware` isn't available with the raw SDK. Agent nodes `authorize_benchmark` (LangGraph `interrupt()` / `Command(resume=…)`, course: ai_orchestration_langgraph) → `enqueue_job` (in-process background task) → `poll_results`. `POST /api/benchmarks`, `GET /api/benchmarks/{job_id}`. UI approval prompt and results table with **measured vs published** labels.
**Acceptance criteria:**
- [ ] No benchmark runs without explicit approval; reject → graph ends cleanly
- [ ] Results show dataset/split, hardware, versions, Recall@k, latency, run ID
**Verification:** `uv run pytest tests/test_benchmark_flow.py -q`; manual: spec example query end to end in UI
**Dependencies:** T10, T14 · **Files:** `agents/graph.py`, `src/api.py`, `ui/src/components/Benchmarks.tsx`, `tests/test_benchmark_flow.py` · **Scope:** M

### Checkpoint D — Full journey
- [ ] Spec example ("CPU-friendly multilingual embedding models … benchmark two") works end to end
- [ ] All tests pass; review with developer

---

## Phase 4: RAG experimentation

### - [ ] T16: Seed corpus + validate/publish + refresh
**Description:** Fill `config/sources.yaml` with ≥100 approved pages from the recommended corpus (D11): Transformers, Datasets, Sentence Transformers and Hugging Face Hub docs, fetched from each project's GitHub docs source at the latest release tag at bootstrap (commit SHA recorded). Verify each repo's license before marking `index_text`; unverified stays `pending_review`. Then batch ingestion, validation (chunk counts, duplicate hashes, orphan URLs, citation resolvability, smoke queries) before flipping the run to `active`; refresh mode re-embeds only changed `content_hash` and tombstones removed chunks.
**Acceptance criteria:**
- [ ] ≥100 approved pages indexed in one active run; every source has license + permitted_use recorded
- [ ] Validation failure leaves the previous active run in place
- [ ] Refresh with no upstream changes writes 0 rows
**Verification:** `uv run python standalone-scripts/ingest.py --all --validate`; `uv run pytest tests/test_ingest_validate.py -q`
**Dependencies:** T5 · **Files:** `src/ingest.py`, `config/sources.yaml`, `standalone-scripts/ingest.py`, `tests/test_ingest_validate.py` · **Scope:** M

### - [ ] T17: Remaining chunkers
**Description:** Add fixed 50-word (rag.ipynb), RecursiveCharacterTextSplitter 200/20 (16_ai_security), Docling HybridChunker max_tokens=128 with the MiniLM tokenizer (advanced_RAG). Each strategy is a separate `ingest_runs` row; retrieval selects it by `run_id`.
**Acceptance criteria:**
- [ ] 4 runs exist over the same corpus revision, one per `chunk_strategy`
- [ ] Every chunk keeps its heading breadcrumb (prepended where the strategy drops it)
**Verification:** `uv run pytest tests/test_chunking.py -q`; SQL count per `chunk_strategy`
**Dependencies:** T16 · **Files:** `src/chunking.py`, `standalone-scripts/ingest.py`, `tests/test_chunking.py` · **Scope:** S

### - [ ] T18: Hybrid retrieval + reranker
**Description:** `retrieval.mode = hybrid` (pgvector + `websearch_to_tsquery`, RRF k=60) and `src/rerank.py` cross-encoder k=10 → 3; both are config switches.
**Acceptance criteria:**
- [ ] Keyword-exact queries (e.g. a parameter name) rank the right chunk higher in hybrid than in dense
- [ ] Reranker on/off changes only ordering/top-n, never which run is queried
**Verification:** `uv run pytest tests/test_retrieval.py -q` (fixture corpus)
**Dependencies:** T6, T16 · **Files:** `src/retrieval.py`, `src/rerank.py`, `tests/test_retrieval.py` · **Scope:** S

### - [ ] T19: Golden dataset in LangSmith
**Description:** ≥120 labelled queries (40 doc Q&A, 30 discovery, 20 constrained comparisons, 10 benchmark, 20 unanswerable/adversarial) with expected source IDs, filters, tool calls; uploaded with `client.create_dataset` (course: llm_observability, `notebooks/golden_qa.json` format).
**Acceptance criteria:**
- [ ] Each example tagged by category; fixed held-out split recorded
- [ ] Dataset is versioned in the repo (`observability/golden/*.json`) and in LangSmith
**Verification:** `uv run python standalone-scripts/upload_golden.py` → dataset count = file count
**Dependencies:** Checkpoint B (can run in parallel) · **Files:** `observability/golden/golden.json`, `standalone-scripts/upload_golden.py` · **Scope:** S

### - [ ] T20: 16-variant eval harness + leaderboard
**Description:** `observability/evals.py` runs 4 chunkers × 2 retrieval × 2 rerank with `client.evaluate`; RAGAS faithfulness/answer relevancy/context precision/recall and openevals judges, both using the same OpenRouter model **through `observability/llm_adapters.py`, which delegates to `src.llm.complete`** (D2a; verify RAGAS/openevals custom-LLM extension points against current docs first); heuristic citation-correctness and Recall@5/MRR; records latency and cost. `GET /api/evals/variants` + UI leaderboard.
**Acceptance criteria:**
- [ ] Judge calls show up as `src.llm` calls (no direct provider client in RAGAS/openevals)
- [ ] 16 experiments with the same dataset and corpus revision; results table with quality, p50/p95 latency, cost
- [ ] Winning config written to `config.yaml` with its experiment IDs
**Verification:** `uv run python standalone-scripts/run_evals.py --variants all`; manual: leaderboard in UI
**Dependencies:** T17, T18, T19 · **Files:** `observability/evals.py`, `observability/llm_adapters.py`, `standalone-scripts/run_evals.py`, `src/api.py`, `ui/src/components/Leaderboard.tsx` · **Scope:** M

### Checkpoint E — Experiments
- [ ] Leaderboard reproducible; measured winner chosen; review with developer

---

## Phase 5: Security

### - [ ] T21: Input guards: guardrails + PII masking (required)
**Description:** Guardrails and PII masking are required deliverables (D3), hand-built because the course middleware isn't available with the raw SDK. `agents/guards.py`: Semantic Router guard with `off_topic` / `potentially_harmful` routes before any LLM call (routing_with_guardrails `SimpleSemanticRouterGuardrail`; encoder + thresholds from config); PII masking for email/credit card (plus phone) on input and before logging/tracing, mirroring `PIIMiddleware` `strategy='mask'`; NeMo Guardrails self-check input via `LLMRails`, with its LLM calls routed through `src.llm.complete` by a custom provider in `agents/nemo_llm.py` (D2a; same model). Wired as the first graph node; emits `guard_events`.
**Acceptance criteria:**
- [ ] Off-topic/harmful prompts are refused with zero synthesis LLM calls
- [ ] Emails/card numbers are masked before reaching the LLM or the logs
- [ ] Block rate and false-positive rate reported on the adversarial split
- [ ] NeMo self-check calls go through `src.llm` (verified by mock in test)
**Verification:** `uv run pytest tests/test_guards.py -q`
**Dependencies:** T10 · **Files:** `agents/guards.py`, `agents/nemo_llm.py`, `agents/graph.py`, `config/nemo/config.yml`, `tests/test_guards.py` · **Scope:** M

### - [ ] T22: Injection defence + redaction + output checks
**Description:** Hardened system prompt with `### RETRIEVED (untrusted) ###` delimiters; retrieval-time redaction of sensitive chunks (16_ai_security `ask_hr_bot_redacted`); output regex check for system-prompt/secret leakage; `html.escape` wherever text could be rendered as HTML.
**Acceptance criteria:**
- [ ] Planted malicious instruction in an indexed test card does not change agent behaviour
- [ ] Output containing the system prompt or a secret pattern is blocked
**Verification:** `uv run pytest tests/test_injection.py -q`
**Dependencies:** T21 · **Files:** `src/prompts.py`, `src/retrieval.py`, `agents/guards.py`, `tests/test_injection.py` · **Scope:** S

### - [ ] T23: Tool safety + red-team report
**Description:** Tool allowlist in the agent; `is_docstring_safe` vetting of MCP tool descriptions at load (16_ai_security); unauthorized category requests blocked by SQL filters. Red-team suite records blocked / allowed / false positive per guard.
**Acceptance criteria:**
- [ ] Tool-escalation cases 100% blocked; unauthorized retrieval = 0; indirect injection ≤1%
- [ ] Report at `observability/reports/redteam.md`
**Verification:** `uv run pytest tests/test_redteam.py -q`
**Dependencies:** T15, T22 · **Files:** `agents/graph.py`, `tools/mcp_server.py`, `tests/test_redteam.py`, `observability/reports/redteam.md` · **Scope:** M

### Checkpoint F — Security
- [ ] Red-team targets met or failures documented; review with developer

---

## Phase 6: Observability + release

### - [ ] T24: LangSmith tracing
**Description:** `observability/tracing.py`: `@traceable` spans for parse, retrieve, rerank, MCP call, generate, guard (llm_observability); tags/metadata `pipeline_variant`, `prompt_version`, `corpus_revision`, `embedding_model_revision`, `model`, `token_usage`, `estimated_cost_usd` (from OpenRouter usage). PII/secret redaction in trace inputs. `src/llm.py` stays free of `src.*` imports (wrapping happens at call sites).
**Acceptance criteria:**
- [ ] ≥99% of golden-set requests produce a complete trace tree
- [ ] No secrets or unmasked PII in traced inputs (checked by test)
**Verification:** `uv run pytest tests/test_tracing.py -q`; manual: LangSmith project shows nested spans
**Dependencies:** T10 · **Files:** `observability/tracing.py`, `agents/graph.py`, `src/retrieval.py`, `tests/test_tracing.py` · **Scope:** M

### - [ ] T25: CI + README + final report
**Description:** GitHub Actions: ruff, pytest (mocked; no RDS/OpenRouter secrets), the `src/llm.py` isolation test, `npm run build`. README covers local run with `uv` + `npm`. Final report with measured scores against spec §7, confidence intervals, failure cases, limitations.
**Acceptance criteria:**
- [ ] CI green on a clean checkout
- [ ] A fresh clone runs the demo from the README alone
- [ ] Report states actual numbers; unmet targets are listed as unmet
**Verification:** CI run passes; manual fresh-clone walkthrough
**Dependencies:** All · **Files:** `.github/workflows/ci.yml`, `README.md`, `observability/reports/final.md` · **Scope:** S

### - [ ] T26: Railway deployment + API auth
**Description:** Deploy API and UI to Railway (no Dockerfile; uv build) **[outside course]**. Auth switched on by `api.auth_enabled: true` in the deployed config, off for the local demo (D9a). Mechanism: **static API key** in an `X-API-Key` header, compared with `secrets.compare_digest` against env var `API_KEY`; the UI sends it from `VITE_API_KEY`. Secrets set as Railway environment variables, never committed.
**Acceptance criteria:**
- [ ] Deployed `/api/*` returns 401 with a missing or wrong key and works with the right one; local run still needs no key
- [ ] Startup fails if `auth_enabled` is true and `API_KEY` is unset
- [ ] Chat, discovery and benchmark smoke tests pass on the deployed instance (RDS reachable, stdio MCP subprocess starts)
**Verification:** `uv run pytest tests/test_auth.py -q`; manual: curl the Railway URL with and without credentials
**Dependencies:** T25 · **Files:** `src/api.py`, `src/auth.py`, `config/config.yaml`, `ui/src/api.ts`, `tests/test_auth.py` · **Scope:** M

### Checkpoint G — Complete
- [ ] Every spec §7 acceptance criterion measured and reported
- [ ] Railway deployment rejects unauthenticated requests
- [ ] Developer sign-off
