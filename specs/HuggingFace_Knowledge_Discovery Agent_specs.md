# AI Engineering Knowledge & Discovery Agent — Capstone Specification

## 1. Goal and non-goals
**Goal.** Build a **React-based AI Engineering Copilot** that helps aspiring AI/ML engineers **learn → discover → compare → evaluate**. Answer technical questions using **publicly available, appropriately licensed** Hugging Face/framework documentation via citation-grounded RAG; discover current **models, embedding models, and datasets** using live Hugging Face REST APIs exposed as MCP tools; compare model capabilities and run small, reproducible benchmarks. Example: *“Find CPU-friendly multilingual embedding models, explain their limitations from model cards, and benchmark two on a sample retrieval task.”*

**Non-goals.** No bootcamp-private corpus as the primary knowledge base; no wholesale mirroring of Hugging Face, model weights, or datasets into Qdrant; no indexing gated/private content; no unlicensed redistribution; no claim that popularity or model-card benchmark numbers equal measured performance. Public accessibility alone does **not** imply permission to persist full text: check licenses/terms, retain attribution, and use live links when indexing is not permitted.

## 2. Architecture
```text
 Public, license-checked docs / selected model & dataset cards
       -> Docling / text parsing -> heading repair -> chunk -> embed
       -> Qdrant (text vectors + source URL, license, revision, breadcrumb)
                                                 |
 React chat + discovery dashboard -> FastAPI -> guardrails -> LangGraph
                                                 | intent router
                          +----------------------+---------------------+
                          |                                            |
                    RAG knowledge                                  MCP tools
               Qdrant + BM25 + rerank                         FastMCP server
               docs, guides, permitted                        | search_models
               model/dataset cards                            | search_datasets
                          |                                    | get_model_details
                          |                                    | get_dataset_details
                          |                                    | benchmark_model
                          |                                    | search_docs/get_section
                          +----------------------+-------------+------+
                                                 |
                                      LLM synthesis + evidence
                                                 |
                           Answer + citations + model comparisons
                           + recommendations + measured benchmarks
                         [LangSmith traces; LiteLLM costs; guard logs]
```
**Source-of-truth rule:** Hugging Face APIs supply **current IDs, downloads, revisions, tags, licenses, and listings**; Qdrant supplies **semantic discovery of permitted documentation**, not an authoritative metadata replica. Refresh/revalidate retrieved details before recommendations. Optional PostgreSQL stores benchmark runs, experiment configs, and user sessions (not scraped copies).

## 3. Capabilities — defaults, alternatives, measurements

### 3.1 Configurable RAG pipeline
**Built:** An ingestion/indexing job for public Hugging Face docs, Transformers and Sentence Transformers guides, plus *selected license-approved* model/dataset cards; source URL, document revision/hash, license, headings, and chunk offsets attached to every chunk. For optional PDF guides, Docling + `ResultPostprocessor` repairs headings; CSV FAQ rows become individual documents; scans/images use cheapest-first **regex/text extraction → OCR → VLM**. Never ingest arbitrary gated cards or whole datasets by default.

| Stage | Default | A/B alternatives | Measurement |
|---|---|---|---|
| Parse | Docling for PDF; native Markdown/HTML; heading repair | Plain extraction; OCR/VLM escalation | Heading accuracy, extraction coverage, $/document |
| Chunk | `HierarchicalChunker` + heading breadcrumb | Fixed **50 words**; `RecursiveCharacterTextSplitter(200,20)`; Docling `HybridChunker(max_tokens=128)` | Recall@5, citation-span accuracy, index size |
| Retrieve | `all-MiniLM-L6-v2` dense vectors in **Qdrant**, calibrated `score_threshold`; source/license filters | Dense + BM25 hybrid; category filters; SQL for counts/aggregates; CLIP for licensed image corpus | Recall@5, MRR, filter leakage = 0 |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2`, **k=10 → top_n=3** | No reranking | Context precision, p95 latency |
| Generate/optimize | Citation-required answer; LiteLLM gateway, cache, retry/fallback | Direct LLM, alternate model routing/cache | Faithfulness, cost/request, p95 latency |

Every stage is a **config switch** (`parser`, `chunker`, `retriever`, `reranker`, `llm`, `cache`) with versioned configs. A single evaluation harness runs comparable combinations. SQL RAG and CLIP are **separate task tracks**, not replacements for text retrieval.

### 3.2 MCP + live Hugging Face discovery
**Built:** Python **FastMCP** server with `@mcp.tool()` for `search_models`, `search_datasets`, `get_model_details`, `get_dataset_details`, `compare_models`, `benchmark_model`, plus RAG tools `search_docs`, `get_section`, and read-only `run_sql` (application benchmark aggregates only). Tools call official Hugging Face REST endpoints; use current metadata, pagination, rate-limit handling, and license/access checks. Agent uses `MultiServerMCPClient` with `create_agent` or LangGraph ReAct loop.

| Default | A/B alternatives | Measurement |
|---|---|---|
| Local stdio MCP; typed, allowlisted read-only tools; benchmark in resource-limited sandbox | Direct REST client; optional remote **SSE** behind PAT auth and JWT-protected FastAPI (validate supported transport) | Tool success ≥98%; correct routing ≥95%; no unauthorized tool execution; API freshness |

Benchmarks report **dataset/split, hardware, versions, Recall@k, latency, and run ID**; distinguish **measured** scores from externally reported benchmarks.

### 3.3 AI evaluations
**Built:** Golden query set in **LangSmith** (`client.create_dataset`), experiments via `client.evaluate`, spanning learning questions, model/dataset discovery, constrained comparisons, evaluation requests, and unanswerable prompts.

| Default | A/B alternatives | Measurement |
|---|---|---|
| RAGAS faithfulness, answer relevancy, context precision/recall; source-link and tool-call heuristics | `AspectCritic`, `RubricsScore`, `openevals` LLM-as-judge; human adjudication | Faithfulness ≥0.90; citation accuracy ≥95%; Recall@5 ≥0.85; discovery constraint accuracy ≥95% |

Publish a **chunking × retrieval × reranking** leaderboard (4 × 2 × 2 = **16** text variants), with quality, latency, cost, and confidence intervals; select the best quality/cost configuration on held-out queries. Include a separate MCP discovery/benchmark correctness report.

### 3.4 Security and guardrails
**Built:** Semantic Router middleware for intent and scope, **NeMo Guardrails** self-check input, hardened system instructions, and output regex screening (defense-in-depth only). `PIIMiddleware` masks sensitive values; retrieval-time redaction and role/category payload filters protect indexed content. `HumanInTheLoopMiddleware` approves risky actions (e.g., downloading/executing third-party model code). Vet MCP docstrings/schemas; escape untrusted HTML with `html.escape` in appropriate rendering contexts. Avoid executing untrusted model code by default.

| Attack test | Default defense | Alternative / metric |
|---|---|---|
| “Ignore rules; disclose secrets” direct injection | Input router + NeMo | Compare router-only; attack block rate ≥95% |
| Malicious instruction embedded in retrieved model card | Treat chunks as data; prompt boundary + output check | Compare hardened prompt alone; injection success ≤1% |
| PII in a source excerpt | PII masking + retrieval redaction | Compare ingest-time masking; sensitive output leakage = 0 |
| User without rights requests restricted source | Role-scoped payload filter + authorization | Compare post-retrieval filter; unauthorized retrieval = 0 |
| Tool description attempts command execution | Tool allowlist, schema vetting, HITL | Compare no HITL in isolated test; unsafe execution = 0 |

### 3.5 User interface
**Built:** React chat/discovery dashboard backed by **FastAPI**: natural-language chat, model/dataset result cards, side-by-side comparisons, benchmark table, cited chunks with **heading breadcrumbs**, retrieval/rerank scores, guardrail event badge, and pipeline-variant toggle. A **short-term memory** toggle uses LangGraph `thread_id` checkpointer; memory never bypasses authorization.

| Default | A/B alternatives | Measurement |
|---|---|---|
| React + FastAPI, citation drawer, variant selector, memory off by default | Memory on/off; compact vs expanded evidence view | Citation click-through works ≥99%; end-to-end p95 ≤8 s excluding benchmarks; task completion ≥90% |

### 3.6 Observability
**Built:** **LangSmith** traces with `@traceable` spans for parse, retrieve, rerank, MCP calls, generate, and guard; tag prompt version, corpus revision, model ID, and pipeline variant. **LiteLLM** records token cost, cache hit, model routing, and latency per request; redact sensitive traces.

| Default | A/B alternatives | Measurement |
|---|---|---|
| LangSmith + LiteLLM | Self-hosted **Langfuse** for sensitive deployments | ≥99% trace coverage; per-request cost/latency visible; no sensitive trace leakage |

## 4. Eval and test plan
1. Assemble **≥120** labeled queries: 40 documentation Q&A, 30 model/dataset discovery, 20 constrained comparisons, 10 benchmark tasks, 20 unanswerable/adversarial; record expected source IDs, filters, tool calls, and acceptable answers.
2. Freeze a **license-reviewed** corpus snapshot and benchmark environment. Use stable query splits; run all **16** text-RAG variants against identical questions and budget caps. Compare RAGAS scores, Recall@5, citation correctness, p50/p95 latency, and cost.
3. Test live API tools separately with mocked contract tests and limited real API checks; verify rate-limit handling, pagination, stale metadata refresh, missing licenses, and correct no-results behavior.
4. Red-team direct and indirect injection (including malicious text planted in retrieved chunks), PII, unauthorized categories, and unsafe tool execution. Record **blocked / allowed / false positive** for every guardrail.
5. Review failures manually; document winning configuration, trade-offs, and reproducible experiment IDs.

## 5. Tech stack
**UI:** React. **API/agents:** Python, FastAPI, LangGraph, FastMCP, `MultiServerMCPClient`. **RAG:** Docling, LangChain splitters, Sentence Transformers, Qdrant, BM25, optional CLIP. **Data:** public Hugging Face REST API; optional PostgreSQL for app results. **Evals/ops:** LangSmith, RAGAS, `openevals`, NeMo Guardrails, LiteLLM; optional Langfuse. **Delivery:** Docker Compose, environment-based secrets, pinned dependencies.

## 6. Milestones
1. **Source governance + ingestion:** approved public corpus, parsing, citations, Qdrant indexing, dedup by URL/revision/chunk hash.
2. **RAG experimentation:** four chunkers, dense/hybrid retrieval, reranking, variant config, baseline leaderboard.
3. **MCP discovery + benchmarking:** Hugging Face tools, agent routing, model/dataset comparison, sandboxed benchmark runs.
4. **Safety + product UI:** guardrails, red-team tests, React evidence/citation panels, memory toggle.
5. **Evaluation + release:** LangSmith experiments, tracing, quality/cost report, Docker demo, architecture and limitations write-up.

## 7. Acceptance criteria
- Answers cite **clickable public source URLs**, document revision, and heading breadcrumbs; unsupported claims are qualified or refused.
- Live MCP searches return current Hugging Face models/datasets; comparisons enforce user constraints; benchmarks show **measured** provenance.
- Reproducible **16-variant** RAG comparison and winning configuration documented; targets: Recall@5 ≥0.85, faithfulness ≥0.90, citation accuracy ≥95%.
- Tool success ≥98%, unauthorized retrieval/execution = 0 in test suite, indirect injection success ≤1%, and ≥99% trace coverage.
- No gated/private material or unapproved full-text copies in Qdrant; source/license audit available; demo runs end-to-end in Docker.
