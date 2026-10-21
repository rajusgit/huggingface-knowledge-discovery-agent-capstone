# Final report: measured results against the spec

Everything below was measured on this repository's code (see the linked files). Where a target was **not met**
or **could not be verified**, it says so. Numbers are not rounded in our favour.

## 1. Acceptance criteria (spec §7)

| Criterion | Target | Measured | Status |
|---|---|---|---|
| Answers cite clickable source URL, revision, heading breadcrumb | yes | Citations carry `source_url`, `revision`, `heading_path`; UI renders them (browser click-through **not** exercised) | Met in code, UI unverified in a browser |
| Unsupported claims qualified or refused | yes | Unanswerable questions refused: 10/10 (agent run 4) and 9/10 (final run); RAG-only variants 100% | Met (one abstention scored as a miss in the last run) |
| Live MCP searches return current Hub data; benchmarks show measured provenance | yes | Discovery: right tool called 30/30, filters/arguments correct 30/30. Every result carries `fetched_at`, `source_url`, revision | Met |
| Comparisons enforce user constraints | yes | `compare_resources` enforces them (unit-tested). But the agent only *passes* the user's constraints in **45%** of golden comparisons | **Partly met** (tool correct, model usage weak) |
| 16-variant comparison reproducible, winner documented | yes | `exp-dc89090e60`, winner `hybrid128 \| dense \| rerank` (see [leaderboard.md](leaderboard.md)) | Met |
| Recall@5 ≥ 0.85 | 0.85 | Winner: **1.00** dev (n=30), **0.90** held-out (n=10) | Met |
| Faithfulness ≥ 0.90 | 0.90 | Winner: **1.00** dev, **1.00** held-out | Met |
| Citation accuracy ≥ 95% | 0.95 | Winner: **0.83** dev, **0.80** held-out | **Not met** (strict metric: a valid citation to a different relevant page counts as a miss) |
| Tool success ≥ 98% | 0.98 | **100%** (102/102 calls in the final run; 107, 114, 117 in earlier runs) | Met |
| Unauthorized retrieval / execution = 0 | 0 | Retrieval leaks 0/4; tool escalation 10/10 blocked; no benchmark ever ran unapproved | Met |
| Indirect injection success ≤ 1% | 1% | **0/44** observed, but the 95% upper bound is **8.0%** (≈380 clean trials would be needed to bound it under 1%) | **Not established** |
| ≥ 99% trace coverage | 99% | Span trees complete and PII-free in a local capture test; **nothing has been sent to or viewed in LangSmith** (no API key) | **Unverified** |
| No gated/private/unapproved text in the store; source/license audit | yes | 152 pages from 5 repos, all verified Apache-2.0 via the GitHub license API; `sources` table + `config/sources.yaml` are the audit trail | Met |
| Demo runs end-to-end in Docker | yes | **No Docker by decision.** Local run documented; CI passes on a clean checkout (macOS); **Railway deployment not performed** | **Not met as written** |

## 2. Other spec targets

| Target | Measured |
|---|---|
| End-to-end p95 ≤ 8 s (excluding benchmarks) | **7.3 s** in the final run (8.2 s in the run before; individual doc questions reached 15 s). Marginal, not comfortable |
| Correct routing ≥ 95% | **Not met.** Agent level: doc 85%, discovery 90%, comparison 85%, benchmark 88%. Router alone: intent 87–88%, docs-vs-tools branch 96–97% |
| Discovery constraint accuracy ≥ 95% | **100%** (30/30) |
| Attack block rate ≥ 95% | Input guard (semantic + LLM check): **9/9** golden attacks, 24/24 on its dev set, **0/124** benign blocked. The semantic layer alone blocks only **67%** on the untouched golden set (96% on the set it was tuned on) |
| Task completion ≥ 90%, citation click-through ≥ 99% | **Not measured** (needs a browser and human tasks) |

## 3. Agent-level evaluation (120 golden items, real agent, sequential)

From [agent_eval.json](agent_eval.json). "Routing" = the router's route is an acceptable one for the category; "grounded" = the expected source was cited or returned by a docs lookup.

| Category | n | Routing | Tool called | Constraints (strict) | Approval gate | Refused | Grounded | p50 / p95 |
|---|---|---|---|---|---|---|---|---|
| doc_qa | 40 | 0.85 | | | | | 0.68 | 3.0 / 8.3 s |
| discovery | 30 | 0.90 | 1.00 | 1.00 | | | | 5.6 / 6.7 s |
| comparison | 20 | 0.85 | 0.85 | 0.45 | | | | 4.1 / 6.5 s |
| benchmark | 10 | 0.88 | | | 0.90 | | | 1.5 / 4.6 s |
| unanswerable | 10 | | | | | 0.90 | | 0.6 / 5.2 s |
| adversarial | 10 | | | | | 1.00 | | 0.0 / 0.7 s |

**The biggest gap:** the RAG component alone retrieves the expected page 100% of the time (Recall@5), but only 68% of doc questions end up grounded in it through the full agent, because some are routed to model discovery. Adding Hub-concept examples to `config/routes.yaml` moved routing 70% → 85% and grounding 57% → 68%; more examples would likely help further.

**Caveat on comparison constraints:** several golden labels encode thresholds the question never states ("CPU-only" → 100M parameters), so strict matching understates the model; key-level matching is no better (45%), so the weak point is real: the model often omits constraints.

## 4. RAG variants (spec §3.1/§4)

- 4 chunkers × dense/hybrid × rerank on/off = 16, all over one corpus revision (`6dddb09d5b55a7a1`, 152 pages). Chunks: hierarchical 5,764; fixed50 5,052; recursive200 14,396; hybrid128 6,844.
- Winner (dev composite 0.943, held-out 0.900): `hybrid128 | dense | rerank`, applied to `config.yaml` and published as the active run.
- **Read the held-out numbers with care:** 10 documentation questions + 2 unanswerable. Differences among the top variants are within noise (e.g. `hierarchical | dense | norerank` scores 0.992 held-out but 0.859 dev). The ranking is a reasonable choice, not a proven one.
- Judges (faithfulness, answer relevancy, context precision/recall) are RAGAS-style but implemented in-house on `src/llm.py`: the `ragas` package failed to import against the current `langchain-community`.
- The run used system prompt v1; prompts were hardened afterwards (v2, tagged in traces). Retrieval-variant ranking is unlikely to hinge on that sentence, but it was not re-run.

## 5. Security (spec §3.4); see [redteam.md](redteam.md)

| Defence | Result |
|---|---|
| Input guard | 9/9 attacks blocked; 0/100 benign blocked |
| Tool escalation | 10/10 blocked, nothing reached the network or the benchmark runner |
| Indirect injection (22 original + 22 later payloads) | 0/44 (upper bound 8.0%) |
| Unauthorized retrieval | 0/4 leaked |
| Retrieved-passage screen | flags 2/180 real corpus passages (1.1%), both legitimate tool-calling docs |

Layers: PII masking before the graph (LangGraph checkpoints its input), semantic guard, LLM self-check, fenced untrusted context, passage screen, access filter, tool allowlist + metadata vetting, human-approved benchmarks, streaming output guard.

## 6. Decisions that differ from the plan

- **No NeMo Guardrails library.** The "self check input" rail is implemented in-house on `src/llm.py` (same technique).
- **No RAGAS/openevals libraries** (see §4).
- **No Docker, no Redis, no LiteLLM, no Qdrant** (per the agreed decisions).
- **Embedding/reranker run on CPU explicitly.** Auto-selected Apple MPS crashed under threads.
- **MCP session is persistent** (one server process): per-call processes re-loaded the models each time.

## 7. Bugs found by measuring (all fixed, each with a regression test)

1. `chunk_id` ignored the run: a second run over unchanged repo commits silently inserted nothing.
2. The cross-encoder returned **NaN on CPU** (memory-mapped weights + CPU BLAS), which silently disabled reranking everywhere until weights were cloned. Reranking now fails loudly on NaN.
3. Concurrent model use crashed the interpreter (segfault, then a Metal assertion).
4. The output guard's 40-character window blocked legitimate answers; now 80, configurable.
5. Any "evaluate" route hijacked ordinary questions into a benchmark prompt; the planner now decides, with a next-best-route fallback.
6. A model outside the allowlist was silently dropped from a benchmark proposal; now it asks.
7. A registry URL checker treated HTTP 429 as "broken" and dropped valid pages.
8. YAML route examples containing `: ` were parsed as mappings.

## 8. Not done, and why

- **Railway deployment and deployed smoke tests**: needs the account owner (T26 is code-complete and tested).
- **LangSmith**: no `LANGSMITH_API_KEY`; the golden-set upload and live traces are untested against the service.
- **Real benchmark dataset**: `standalone-scripts/huggingface_seeding_script.py` fetches model *metadata* (downloads, likes) as documents, which the spec says not to store; it does not produce corpus/queries/relevance judgements. Benchmarks currently run on a 12-document fixture.
- **Browser verification of the UI**, Linux CI run (the CPU-torch wheel path), streaming token cost.

## 9. Known limitations

- A static API key built into the UI bundle is visible to anyone who can load the UI.
- The golden set is author-written; the input guard's examples and the router's utterances were tuned with knowledge of earlier failures, so those scores are optimistic.
- Hub rate limits: 429s are retried and surfaced as "unverified", never as broken.
- Citation correctness (0.83) and routing (85–90%) are the weakest measured quality dimensions.
