export interface Citation {
  n: number
  chunk_id: string
  source_url: string
  heading_path: string[]
  revision: string
  content: string
  score: number
}

export interface LiveSource {
  tool: string
  source_url: string
  fetched_at: string | null
}

export interface BenchmarkRecord {
  job_id: string
  model_id: string
  model_revision: string | null
  dataset_id: string
  dataset_revision: string | null
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'timeout'
  seed: number | null
  hardware: Record<string, string | number> | null
  metrics: {
    'recall@1'?: number; 'recall@5'?: number; 'recall@10'?: number; 'mrr@10'?: number; 'ndcg@10'?: number
    n_docs?: number; n_queries?: number; peak_rss_mb?: number | null
    timing?: { docs_per_second?: number | null; query_latency_ms?: number | null }
  } | null
  error: string | null
}

export interface Approval {
  kind: string
  models: string[]
  dataset: string
  seed: number
  limits: { max_runtime_seconds: number; max_memory_mb: number }
}

export type ChatEvent =
  | { type: 'node'; name: string }
  | ({ type: 'approval_required' } & Approval)
  | { type: 'benchmark_status'; jobs: { job_id: string; model_id: string; status: string }[] }
  | { type: 'token'; text: string }
  | {
      type: 'final'
      answer: string
      routes: string[]
      citations: Citation[]
      invalid_citations: number[]
      live_sources: LiveSource[]
      benchmarks: BenchmarkRecord[]
      guard_events: unknown[]
    }
  | { type: 'error'; message: string }

export const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

/** X-API-Key for a deployed API (the key is embedded in the bundle at build time: visible to anyone who can load the UI). */
export function authHeaders(): Record<string, string> {
  const key = import.meta.env.VITE_API_KEY as string | undefined
  return key ? { 'X-API-Key': key } : {}
}

/** Split a growing SSE buffer into complete `data:` events plus the unfinished tail. */
export function parseSSE(rest: string, chunk: string): { events: ChatEvent[]; rest: string } {
  const parts = (rest + chunk).split('\n\n')
  const tail = parts.pop() ?? ''
  const events: ChatEvent[] = []
  for (const part of parts) {
    const line = part.split('\n').find((l) => l.startsWith('data: '))
    if (line) events.push(JSON.parse(line.slice(6)) as ChatEvent)
  }
  return { events, rest: tail }
}

/** Only http(s) links are ever rendered as hrefs. */
export function safeUrl(url: string): string | null {
  try {
    const u = new URL(url)
    return u.protocol === 'https:' || u.protocol === 'http:' ? u.href : null
  } catch {
    return null
  }
}

export type Piece = { text: string } | { n: number }

/** Split answer text into plain text and citation markers; [1, 6] becomes two markers. */
export function splitCitations(text: string): Piece[] {
  const pieces: Piece[] = []
  for (const s of text.split(/(\[\d+(?:\s*,\s*\d+)*\])/).filter(Boolean)) {
    const m = /^\[(\d+(?:\s*,\s*\d+)*)\]$/.exec(s)
    if (m) for (const n of m[1].split(',')) pieces.push({ n: Number(n.trim()) })
    else pieces.push({ text: s })
  }
  return pieces
}

/** One stable thread while memory is on; a fresh thread per message when it is off. */
export function threadIdFor(memory: boolean, stable: string): string {
  return memory ? stable : crypto.randomUUID()
}

export interface SearchFilters {
  q?: string
  task?: string
  library?: string
  language?: string
  license?: string
  sort?: string
  limit?: number
}
export type HubKind = 'models' | 'datasets'

// Filters each endpoint accepts (see src/api.py); anything else is never sent.
const FILTERS: Record<HubKind, (keyof SearchFilters)[]> = {
  models: ['q', 'task', 'library', 'language', 'license', 'sort', 'limit'],
  datasets: ['q', 'language', 'license', 'sort', 'limit'],
}

export function buildSearchUrl(kind: HubKind, f: SearchFilters): string {
  const p = new URLSearchParams()
  for (const k of FILTERS[kind]) {
    const v = f[k]
    if (v !== undefined && v !== '') p.set(k, String(v))
  }
  const qs = p.toString()
  return `${API_URL}/api/${kind}/search${qs ? `?${qs}` : ''}`
}

export interface HubResult {
  id: string
  downloads: number | null
  likes: number | null
  pipeline_tag?: string | null
  library_name?: string | null
  license: string | null
  last_modified: string | null
  gated: boolean | null
  tags: string[]
  source_url: string
}
export interface HubResponse {
  results: HubResult[]
  fetched_at: string
  time_sensitive_fields: string[]
}

export async function searchHub(kind: HubKind, f: SearchFilters, signal?: AbortSignal): Promise<HubResponse> {
  const res = await fetch(buildSearchUrl(kind, f), { headers: authHeaders(), signal })
  if (!res.ok) {
    const detail = res.status === 422 ? 'Invalid filters.' : res.status === 502 ? 'Hugging Face is unavailable.' : `Request failed (${res.status}).`
    throw new Error(detail)
  }
  return (await res.json()) as HubResponse
}

export interface MetricSummary { mean: number | null; ci_low: number | null; ci_high: number | null; n: number }
export interface VariantRow {
  variant: string
  dev: Record<string, MetricSummary | number | null>
  heldout: Record<string, MetricSummary | number | null>
  latency_p50_ms: number
  latency_p95_ms: number
  retrieval_p50_ms: number
  retrieval_p95_ms: number
  cost_per_query: number
}
export interface LeaderboardData {
  experiment_id?: string
  winner?: { variant: string } | null
  variants: VariantRow[]
  message?: string
}

export function leaderboardRows(d: LeaderboardData) {
  const num = (v: unknown) => (typeof v === 'number' ? v : null)
  const f3 = (v: number | null) => (v === null ? '—' : v.toFixed(3))
  return [...d.variants]
    .sort((a, b) => (num(b.dev.composite) ?? -1) - (num(a.dev.composite) ?? -1))
    .map((r) => {
      const rec = r.dev['recall@5'] as MetricSummary | undefined
      return {
        variant: r.variant,
        isWinner: d.winner?.variant === r.variant,
        devComposite: f3(num(r.dev.composite)),
        heldoutComposite: f3(num(r.heldout.composite)),
        recall: rec?.mean == null ? '—' : `${rec.mean.toFixed(2)} [${rec.ci_low!.toFixed(2)}, ${rec.ci_high!.toFixed(2)}]`,
        p50: Math.round(r.retrieval_p50_ms),
        p95: Math.round(r.retrieval_p95_ms),
        e2eP95: Math.round(r.latency_p95_ms),
        cost: r.cost_per_query.toFixed(5),
      }
    })
}

export async function fetchLeaderboard(signal?: AbortSignal): Promise<LeaderboardData> {
  const res = await fetch(`${API_URL}/api/evals/variants`, { headers: authHeaders(), signal })
  if (!res.ok) throw new Error(`Request failed (${res.status})`)
  return (await res.json()) as LeaderboardData
}

export const fmtMetric = (v: number | null | undefined): string => (v === null || v === undefined ? '—' : v.toFixed(3))

/** Table rows (label/value) for one benchmark run: quality, speed, then provenance. */
export function benchmarkRows(r: BenchmarkRecord): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = [{ label: 'Status', value: r.status }]
  if (r.error) rows.push({ label: 'Error', value: r.error })
  const m = r.metrics
  if (!m) return rows
  const hw = r.hardware ?? {}
  return [
    ...rows,
    { label: 'Recall@1', value: fmtMetric(m['recall@1']) },
    { label: 'Recall@5', value: fmtMetric(m['recall@5']) },
    { label: 'Recall@10', value: fmtMetric(m['recall@10']) },
    { label: 'MRR@10', value: fmtMetric(m['mrr@10']) },
    { label: 'nDCG@10', value: fmtMetric(m['ndcg@10']) },
    { label: 'Docs / second', value: m.timing?.docs_per_second != null ? m.timing.docs_per_second.toFixed(1) : '—' },
    { label: 'Query latency (ms)', value: m.timing?.query_latency_ms != null ? m.timing.query_latency_ms.toFixed(1) : '—' },
    { label: 'Peak memory (MB)', value: m.peak_rss_mb != null ? String(Math.round(m.peak_rss_mb)) : '—' },
    { label: 'Corpus / queries', value: `${m.n_docs ?? '—'} docs / ${m.n_queries ?? '—'} queries` },
    { label: 'Model revision', value: (r.model_revision ?? '—').slice(0, 8) },
    { label: 'Dataset', value: `${r.dataset_id} @ ${r.dataset_revision ?? '—'}` },
    { label: 'Seed', value: String(r.seed ?? '—') },
    { label: 'Hardware', value: `${hw.cpu_count ?? '?'} CPU · ${hw.platform ?? '?'} · torch ${hw.torch ?? '?'} · sentence-transformers ${hw.sentence_transformers ?? '?'}` },
  ]
}

export function parseIds(text: string): string[] {
  return [...new Set(text.split(/[\s,]+/).map((s) => s.trim()).filter(Boolean))]
}

export interface ConstraintForm {
  licenses: string
  maxParamsM: string
  language: string
  task: string
}
export interface Constraints {
  licenses?: string[]
  max_parameters?: number
  language?: string
  task?: string
}

export function buildConstraints(f: ConstraintForm): Constraints | undefined {
  const c: Constraints = {}
  const licenses = f.licenses.split(',').map((s) => s.trim()).filter(Boolean)
  if (licenses.length) c.licenses = licenses
  const m = Number(f.maxParamsM)
  if (f.maxParamsM.trim() !== '' && Number.isFinite(m) && m > 0) c.max_parameters = Math.round(m * 1_000_000)
  if (f.language.trim()) c.language = f.language.trim()
  if (f.task.trim()) c.task = f.task.trim()
  return Object.keys(c).length ? c : undefined
}

export interface CompareRow {
  id: string
  pipeline_tag: string | null
  library_name: string | null
  license: string | null
  parameters: number | null
  languages: string[] | null
  downloads: number | null
  likes: number | null
  revision: string | null
  gated: boolean | null
  source_url: string
  fetched_at: string
}
export interface CompareResponse {
  models: CompareRow[]
  warnings: { id: string; field: string; message: string }[]
  fetched_at: string
  time_sensitive_fields: string[]
  constraints_check?: Record<string, { ok: boolean; violations: string[]; unknown: string[] }>
}

export async function compareResources(
  ids: string[],
  kind: HubKind,
  constraints?: Constraints,
  signal?: AbortSignal,
): Promise<CompareResponse> {
  const res = await fetch(`${API_URL}/api/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ ids, kind, constraints }),
    signal,
  })
  if (!res.ok) {
    throw new Error(res.status === 422 ? 'Enter 2-5 valid, distinct ids (owner/name) and valid constraints.' : res.status === 502 ? 'Hugging Face is unavailable.' : `Request failed (${res.status}).`)
  }
  return (await res.json()) as CompareResponse
}

async function* readEvents(res: Response): AsyncGenerator<ChatEvent> {
  if (!res.ok || !res.body) throw new Error(`Request failed (${res.status})`)
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let rest = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    const parsed = parseSSE(rest, decoder.decode(value, { stream: true }))
    rest = parsed.rest
    for (const e of parsed.events) yield e
  }
}

const post = (path: string, body: unknown, signal?: AbortSignal) =>
  fetch(`${API_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body),
    signal,
  })

export async function* streamChat(
  message: string,
  threadId: string,
  memory: boolean,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  yield* readEvents(await post('/api/chat', { message, thread_id: threadId, pipeline_variant: 'default', memory }, signal))
}

/** Answer a pending approval (e.g. run the proposed benchmark). Uses the same thread as the request. */
export async function* resumeChat(
  threadId: string,
  approved: boolean,
  memory: boolean,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  yield* readEvents(await post('/api/chat/resume', { thread_id: threadId, approved, memory }, signal))
}
