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

export type ChatEvent =
  | { type: 'node'; name: string }
  | { type: 'token'; text: string }
  | {
      type: 'final'
      answer: string
      routes: string[]
      citations: Citation[]
      invalid_citations: number[]
      live_sources: LiveSource[]
      guard_events: unknown[]
    }
  | { type: 'error'; message: string }

export const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

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
  const res = await fetch(buildSearchUrl(kind, f), { signal })
  if (!res.ok) {
    const detail = res.status === 422 ? 'Invalid filters.' : res.status === 502 ? 'Hugging Face is unavailable.' : `Request failed (${res.status}).`
    throw new Error(detail)
  }
  return (await res.json()) as HubResponse
}

export async function* streamChat(
  message: string,
  threadId: string,
  memory: boolean,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, thread_id: threadId, pipeline_variant: 'default', memory }),
    signal,
  })
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
