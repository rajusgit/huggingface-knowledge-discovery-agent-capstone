# Implementation Plan: Hugging Face Knowledge & Discovery Agent (Capstone)

## Overview

A chat-and-discovery copilot for aspiring AI/ML engineers: **Learn → Discover → Compare → Evaluate**. It answers questions from license-checked public Hugging Face docs with cited RAG (Postgres + pgvector), finds live models and datasets through FastMCP tools that wrap the Hugging Face Hub API, compares them against user constraints, and runs small reproducible embedding benchmarks. It uses React + FastAPI + LangGraph, with OpenRouter as the only LLM provider.

Sources: `specs/HuggingFace_Knowledge_Discovery Agent_specs.md` (product), `specs/HuggingFace_Knowledge_Discovery Agent_technical_architecture.md` (architecture), and the bootcamp course material in `../ai-engineering-bootcamp-codebasics` (`course-summary.html`, notebooks, session repos). Every technique in this plan is mapped to the course notebook it comes from; the few items the course doesn't cover are marked **[outside course]**.

## Architecture Decisions (developer-confirmed)

| # | Decision | Course anchor |
|---|---|---|
| D1 | **Layout:** core libraries in `src/` (config, db, embed, llm, ingest, chunking, retrieval, rerank); `agents/` = LangGraph graph; `tools/` = FastMCP server + HF client + benchmark runner; `observability/` = tracing + evals; `ui/` = React app; `standalone-scripts/` = one-off CLIs (ingest, ask, eval runs). | github-agent `src/agents/graph.py`, hr-assist `server.py` |
| D2 | **LLM:** OpenRouter only, `google/gemini-3.5-flash-lite`, called through the **raw `openai` SDK** in `src/llm.py`, which imports no `src.*` module. No LiteLLM, no LangChain chat model. | 16_llm_gateways (retries/fallbacks idea) **[OpenRouter outside course]** |
| D2a | **Single LLM gateway:** every LLM call in the project goes through `src/llm.py`, the agent's and also those made by NeMo Guardrails, RAGAS and openevals. Those libraries get thin adapters that call `src.llm.complete`. Same OpenRouter model everywhere. A test fails if any other module imports `openai` or creates an LLM client. | — |
| D3 | **No LangChain agent middleware** (follows from D2): routing, caching and the three required safety features are hand-built as plain functions / LangGraph nodes. **PII masking, human approval and guardrails are required deliverables** (T15, T21, T22), not optional. | react_langgraph (explicit loop), ai_orchestration_langgraph (`interrupt()`), routing_with_guardrails, 16_ai_security |
| D3a | **Everything tunable lives in `config.yaml`:** LLM model, embedding model, Semantic Router encoder (MiniLM for now), reranker, thresholds, k/top_n, chunk sizes, timeouts, retries, TTLs, benchmark limits. No model name or tunable is hard-coded in Python; a test enforces this. | — |
| D3b | **Retries:** 3 retries with backoff before erroring out, for both LLM and HF API calls (`max_retries: 3`). Other limits use defaults (below) and stay configurable. | 16_llm_gateways `num_retries` |
| D4 | **Vector store:** Postgres + pgvector + Postgres full-text (RRF fusion), new schema `hf_discovery` on the existing RDS; no existing tables reused. | embeddings_and_vectordb, advanced_RAG (hybrid idea) **[pgvector outside course; course uses Qdrant]** |
| D5 | **Embeddings:** `all-MiniLM-L6-v2` (384-dim) via `src/embed.py`. Reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, k=10 → 3. | Session 2/3, advanced_RAG |
| D6 | **Config:** `config/config.yaml` (non-secrets) + `.env` (secrets). | — |
| D7 | **Memory:** LangGraph **Postgres checkpointer** in `hf_discovery`, keyed by `thread_id`. | Session 6/7 checkpointer + thread_id **[Postgres saver outside course; course uses Sqlite/InMemory]** |
| D8 | **Cache:** no Redis. Exact-match in-process LLM cache inside `src/llm.py` (raw SDK, so LangChain `InMemoryCache` doesn't apply) and a TTL cache for HF API calls. | 16_advanced_ai_eng exact-match cache |
| D9 | **Delivery:** no Docker at all; run locally with `uv` (course pattern). CI (GitHub Actions) kept. Hosted deploy target is **Railway** (built without a Dockerfile). | course repos use `pyproject.toml` + `uv.lock` **[Railway outside course]** |
| D9a | **Auth:** none for the local demo; a **static API key** required when deployed to Railway. Controlled by a config switch (`api.auth_enabled`); the key comes from env var `API_KEY`. | hr-assist-remote (PAT-style token check) |
| D10 | **UI:** React + TypeScript (Vite) talking to FastAPI. | mcp-client `fastapi_server.py` (FastAPI + CORS) **[React outside course]** |
| D11 | **Seed corpus (recommended):** official docs of Transformers, Datasets, Sentence Transformers and the Hugging Face Hub, fetched from each project's GitHub docs source at the **latest release tag at bootstrap time** (commit SHA recorded per source). License verified per repo before indexing; anything unverified stays `pending_review`. Registry lives in `config/sources.yaml`. | evals_with_ragas / advanced_RAG ingestion patterns |
| D12 | **Benchmark dataset:** comes from a standalone fetch script **the developer will provide** (goes in `standalone-scripts/`); the script covers the benchmark dataset only, not the docs corpus. Until then, tests use a tiny in-repo fixture. | — |

**Config defaults (Q4; all overridable in `config.yaml`):**

| Key | Default |
|---|---|
| `llm.openrouter.max_retries` / `huggingface.max_retries` | **3** (confirmed) |
| `llm.openrouter.timeout_seconds` | 60 |
| `huggingface.api_timeout_seconds` | 30 |
| `huggingface.cache_ttl_seconds` | 900 (15 min, low end of the spec's 15–60 min) |
| `llm.cache_enabled` | true (exact-match, `temperature=0` only) |
| `benchmarks.max_runtime_seconds` | 600 |
| `benchmarks.max_memory_mb` | 4096 |
| `router.encoder` | `all-MiniLM-L6-v2` |

## Dependency Graph

```
config.yaml + src/config.py ──┬── src/db.py + schema.sql ──┬── src/ingest.py + chunking.py ──┐
                              │                            │                                 │
                              ├── src/embed.py ────────────┴── src/retrieval.py (+ rerank) ──┤
                              │                                                              │
src/llm.py (standalone) ──────┼──────────────────────────────────────────────────────────────┤
                              │                                                              ▼
                              └── tools/hf_client.py ── tools/mcp_server.py ──────► agents/graph.py
                                                         tools/benchmarks.py          (LangGraph +
                                                                                      PG checkpointer)
                                                                                           │
                                          observability/ (tracing, evals) ◄────────────────┤
                                                                                           ▼
                                                                                  src/api.py (FastAPI)
                                                                                           │
                                                                                           ▼
                                                                                   ui/ (React)
```

Build order is bottom-up, but sliced **vertically**: Phase 1 ships one thin Learn path (one doc → cited answer → API → UI) before widening anything.

## Task List

Full task details (acceptance criteria, verification, files) are in `tasks/todo.md`.

### Phase 0: Foundation
- [x] T1: Align architecture spec with decisions D1–D10
- [x] T2: Project skeleton + config (pyproject/uv, config.yaml, src/config.py, .env.example)
- [x] T3: `hf_discovery` schema + db init on RDS
- [x] T4: `src/llm.py` OpenRouter client (complete/stream/cache) + isolation test

**Checkpoint A — Foundation:** config loads, schema exists on RDS, one live OpenRouter call works, tests pass.

### Phase 1: Thin Learn slice (end-to-end)
- [x] T5: Ingest one approved doc page → hierarchical chunks → active run
- [ ] T6: Dense retrieval + cited answer CLI
- [ ] T7: FastAPI `/api/chat` (RAG-only, SSE streaming)
- [ ] T8: React chat shell with citation drawer

**Checkpoint B — Learn path:** question in browser → streamed answer with clickable source URL + breadcrumb.

### Phase 2: Discover (MCP + agent)
- [ ] T9: HF client + FastMCP server (search/details tools)
- [ ] T10: LangGraph agent: intent routing (Semantic Router), RAG and MCP branches, Postgres checkpointer
- [ ] T11: Discovery endpoints + agent-backed `/api/chat`
- [ ] T12: UI discovery cards, filters, memory toggle

**Checkpoint C — Discover path:** "find CPU-friendly multilingual embedding models" returns live HF results with `fetched_at`; follow-ups use thread memory.

### Phase 3: Compare + Evaluate (benchmarks)
- [ ] T13: `get_card_section` + `compare_resources` tools, `/api/compare`, UI compare table
- [ ] T14: Benchmark runner (sandboxed subprocess) + `benchmark_runs` + MCP tool
- [ ] T15: Benchmark approval (LangGraph `interrupt()`), `/api/benchmarks`, UI results table

**Checkpoint D — Full journey:** the spec's example query works end to end: discovery, card limitations, and an approved benchmark of two models with measured provenance.

### Phase 4: RAG experimentation
- [ ] T16: Seed corpus (≥100 approved pages), validate/publish, refresh by content hash
- [ ] T17: Remaining 3 chunkers as separate ingest runs
- [ ] T18: Hybrid retrieval (pgvector + FTS, RRF) + reranker switches
- [ ] T19: Golden dataset (≥120 queries) in LangSmith
- [ ] T20: 16-variant eval harness (RAGAS + heuristics) + leaderboard endpoint/UI

**Checkpoint E — Experiments:** 16-row leaderboard reproducible; best measured config recorded in config.yaml.

### Phase 5: Security
- [ ] T21: Input guards: Semantic Router guard, PII masking, NeMo self-check input
- [ ] T22: Injection defence, retrieval-time redaction, output checks, `html.escape`
- [ ] T23: Tool safety (allowlist, docstring vetting) + red-team suite and report

**Checkpoint F — Security:** red-team suite: tool escalation 100% blocked, indirect injection ≤1%, unauthorized retrieval = 0.

### Phase 6: Observability + release
- [ ] T24: LangSmith `@traceable` spans, tags/metadata, OpenRouter cost
- [ ] T25: CI (GitHub Actions) + README local run + final quality/cost report
- [ ] T26: Railway deployment with API auth enabled

**Checkpoint G — Complete:** all spec §7 acceptance criteria measured and reported (met or not, with numbers); Railway deploy rejects unauthenticated requests.

## Parallelization

- After Checkpoint A: T5–T6 (RAG) and T9 (HF/MCP) are independent.
- After T7's API contract is fixed: T8 (UI) can run in parallel with T9/T10.
- T19 (golden dataset authoring) can start any time after Checkpoint B.
- Must be sequential: T3 (schema) before any DB task; T10 before T11/T15; T16–T18 before T20.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| RDS user can't `CREATE EXTENSION vector` / `CREATE SCHEMA` | High | T3 runs first in Phase 0; stop and escalate if it fails |
| pgvector < 0.5 (no HNSW) | Med | T3 prints the version; fall back to IVFFlat |
| Raw SDK + no middleware: course examples (PIIMiddleware, HITL, GuardrailsMiddleware, `create_agent`) can't be copied directly | Med | Hand-build as LangGraph nodes (T10, T15, T21); keep each one tiny and tested |
| NeMo Guardrails, RAGAS and openevals call an LLM themselves, which would bypass `src/llm.py` (D2a) | Med | Adapters that delegate to `src.llm.complete` (NeMo custom LLM provider, RAGAS/openevals custom LLM wrappers). Verify each library's extension point against its current docs in T20/T21; if one can't be adapted, stop and ask |
| Benchmark dataset script not yet available (D12) | Med | T14 builds the runner against a fixture; the real dataset plugs in when the script arrives |
| Railway runtime differs from local (stdio MCP subprocess, model downloads, RDS reachability) | Med | T26 smoke-tests chat, discovery and benchmark on the deployed instance |
| `MultiServerMCPClient` returns LangChain tools, but the agent uses the raw SDK | Low | Convert tool schemas with `convert_to_openai_tool`, call tools through the adapter |
| Only one `active` ingest run (schema unique index), but experiments need 4 chunker runs queryable at once | Med | Retrieval takes an optional `run_id`; "active" is only the production default (T17) |
| No Docker, so the spec's "restricted containers" for benchmarks aren't available | Med | Subprocess with timeout + `resource` limits + allowlisted models/datasets, no `trust_remote_code` (T14) |
| HF API rate limits / schema drift | Med | Backoff, TTL cache, mocked contract tests + a few live checks (T9) |
| License review of ~100 pages is slow, manual work | Med | Start registry curation early (parallel with Phase 1–3) |
| Targets (Recall@5 ≥0.85, faithfulness ≥0.90) may not be met | Low | Report measured numbers with failure analysis; don't claim unmeasured targets |

## Resolved Questions (developer answers)

- **Q1:** Rename `src/tools.py` → `src/retrieval.py`. Agreed.
- **Q2:** Same OpenRouter model; all LLM calls, the libraries' included, route through `src/llm.py` (D2a).
- **Q3:** Semantic Router uses MiniLM for now; every model and tunable is configurable in `config.yaml` (D3a).
- **Q4:** 3 retries before erroring out; other limits use defaults and stay configurable (D3b, config defaults table).
- **Q5:** Use the recommended seed corpus (D11).
- **Q6:** Developer will provide a standalone fetch script later (D12).
- **Q7:** No auth locally; auth required on Railway (D9a, T26).

- **Q8:** Railway auth uses a **static API key** (header checked by FastAPI; key in a Railway env var, never in config.yaml).
- **Q9:** The developer's script fetches the **benchmark dataset only**; the seed docs corpus is fetched by our own ingestion (T16).

## Open Questions

None at present.
