import { describe, expect, it } from 'vitest'
import { parseSSE, safeUrl, splitCitations } from './api'

describe('parseSSE', () => {
  it('parses complete events and keeps a partial tail for the next chunk', () => {
    const a = parseSSE('', 'data: {"type":"token","text":"Hi"}\n\ndata: {"type":"tok')
    expect(a.events).toEqual([{ type: 'token', text: 'Hi' }])
    const b = parseSSE(a.rest, 'en","text":"!"}\n\n')
    expect(b.events).toEqual([{ type: 'token', text: '!' }])
    expect(b.rest).toBe('')
  })
  it('ignores non-data lines', () => {
    expect(parseSSE('', ': ping\n\n').events).toEqual([])
  })
})

describe('safeUrl', () => {
  it('allows only http(s)', () => {
    expect(safeUrl('https://huggingface.co/docs')).toBe('https://huggingface.co/docs')
    expect(safeUrl('javascript:alert(1)')).toBeNull()
    expect(safeUrl('not a url')).toBeNull()
  })
})

describe('splitCitations', () => {
  it('splits text around [n] markers', () => {
    expect(splitCitations('a [1] b [2].')).toEqual([
      { text: 'a ' }, { n: 1 }, { text: ' b ' }, { n: 2 }, { text: '.' },
    ])
  })
})

describe('splitCitations with comma lists', () => {
  it('expands [1, 6] into separate markers', () => {
    expect(splitCitations('see [1, 6].')).toEqual([{ text: 'see ' }, { n: 1 }, { n: 6 }, { text: '.' }])
  })
})

import { buildSearchUrl, threadIdFor } from './api'

describe('buildSearchUrl', () => {
  it('maps filters 1:1 to API params and omits empty ones', () => {
    const u = new URL(buildSearchUrl('models', { q: 'multi', task: 'sentence-similarity', library: '', language: 'fr', sort: 'likes', limit: 5 }), 'http://x')
    expect(u.pathname).toBe('/api/models/search')
    expect(Object.fromEntries(u.searchParams)).toEqual({ q: 'multi', task: 'sentence-similarity', language: 'fr', sort: 'likes', limit: '5' })
  })
  it('datasets only send the filters the datasets API supports', () => {
    const u = new URL(buildSearchUrl('datasets', { q: 'squad', task: 'qa', library: 'x', language: 'en', license: 'mit' }), 'http://x')
    expect(u.pathname).toBe('/api/datasets/search')
    expect(Object.fromEntries(u.searchParams)).toEqual({ q: 'squad', language: 'en', license: 'mit' })
  })
  it('encodes values', () => {
    expect(buildSearchUrl('models', { q: 'a&b=c' })).toContain('q=a%26b%3Dc')
  })
})

describe('threadIdFor', () => {
  it('keeps one thread while memory is on', () => {
    expect(threadIdFor(true, 'stable')).toBe('stable')
  })
  it('uses a fresh thread for every message while memory is off', () => {
    expect(threadIdFor(false, 'stable')).not.toBe('stable')
    expect(threadIdFor(false, 'stable')).not.toBe(threadIdFor(false, 'stable'))
  })
})

import { buildConstraints, parseIds } from './api'

describe('parseIds', () => {
  it('splits on commas, spaces and newlines, trims and de-duplicates', () => {
    expect(parseIds('a/b, c/d\n a/b  e/f')).toEqual(['a/b', 'c/d', 'e/f'])
    expect(parseIds('  ')).toEqual([])
  })
})

describe('buildConstraints', () => {
  it('maps form fields to the API constraint keys and drops empties', () => {
    expect(buildConstraints({ licenses: 'mit, Apache-2.0', maxParamsM: '150', language: ' fr ', task: '' })).toEqual({
      licenses: ['mit', 'Apache-2.0'],
      max_parameters: 150_000_000,
      language: 'fr',
    })
    expect(buildConstraints({ licenses: '', maxParamsM: '', language: '', task: '' })).toBeUndefined()
  })
  it('ignores a non-numeric parameter budget', () => {
    expect(buildConstraints({ licenses: '', maxParamsM: 'abc', language: 'en', task: '' })).toEqual({ language: 'en' })
  })
})

import { benchmarkRows, fmtMetric, type BenchmarkRecord } from './api'

const rec = (over: Partial<BenchmarkRecord> = {}): BenchmarkRecord => ({
  job_id: 'j1', model_id: 'o/m', model_revision: '1110a243fdf4706b3f48f1d95db1a4f5529b4d41', dataset_id: 'd', dataset_revision: 'fx1',
  status: 'succeeded', seed: 0, hardware: { cpu_count: 8, platform: 'macOS', torch: '2.0', sentence_transformers: '6.1.0' },
  metrics: { 'recall@1': 0.5, 'recall@5': 0.9, 'recall@10': 1, 'mrr@10': 0.8, 'ndcg@10': 0.85, n_docs: 12, n_queries: 6,
    peak_rss_mb: 543, timing: { docs_per_second: 448.14, query_latency_ms: 5.343 } }, error: null, ...over,
})

describe('fmtMetric', () => {
  it('formats scores to 3 decimals and handles missing values', () => {
    expect(fmtMetric(0.91234)).toBe('0.912')
    expect(fmtMetric(1)).toBe('1.000')
    expect(fmtMetric(undefined)).toBe('—')
    expect(fmtMetric(null)).toBe('—')
  })
})

describe('benchmarkRows', () => {
  it('lists quality, speed and provenance rows for a successful run', () => {
    const rows = Object.fromEntries(benchmarkRows(rec()).map((r) => [r.label, r.value]))
    expect(rows['Recall@5']).toBe('0.900')
    expect(rows['nDCG@10']).toBe('0.850')
    expect(rows['Docs / second']).toBe('448.1')
    expect(rows['Model revision']).toBe('1110a243')
    expect(rows['Dataset']).toBe('d @ fx1')
    expect(rows['Hardware']).toContain('8 CPU')
  })
  it('shows only status and error for a failed run', () => {
    const rows = benchmarkRows(rec({ status: 'timeout', metrics: null, error: 'exceeded the 600s runtime limit' }))
    expect(rows.map((r) => r.label)).toEqual(['Status', 'Error'])
    expect(rows[1].value).toContain('600s')
  })
})

import { leaderboardRows, type LeaderboardData } from './api'

const lb = (): LeaderboardData => ({
  experiment_id: 'exp-1',
  winner: { variant: 'b|hybrid|rerank' },
  variants: [
    { variant: 'a|dense|norerank', dev: { composite: 0.5, 'recall@5': { mean: 0.6, ci_low: 0.4, ci_high: 0.8, n: 10 } }, heldout: { composite: 0.4 }, latency_p50_ms: 100, latency_p95_ms: 200, retrieval_p50_ms: 10, retrieval_p95_ms: 20, cost_per_query: 0.001 },
    { variant: 'b|hybrid|rerank', dev: { composite: 0.9, 'recall@5': { mean: 0.95, ci_low: 0.8, ci_high: 1, n: 10 } }, heldout: { composite: 0.85 }, latency_p50_ms: 300, latency_p95_ms: 500, retrieval_p50_ms: 30, retrieval_p95_ms: 50, cost_per_query: 0.002 },
  ],
})

describe('leaderboardRows', () => {
  it('sorts by dev composite, flags the winner and formats cells', () => {
    const rows = leaderboardRows(lb())
    expect(rows.map((r) => r.variant)).toEqual(['b|hybrid|rerank', 'a|dense|norerank'])
    expect(rows[0].isWinner).toBe(true)
    expect(rows[0].recall).toBe('0.95 [0.80, 1.00]')
    expect(rows[0].devComposite).toBe('0.900')
    expect(rows[1].cost).toBe('0.00100')
  })
  it('shows a dash for missing scores', () => {
    const d = lb()
    d.variants[0].heldout.composite = null
    expect(leaderboardRows(d).find((r) => r.variant.startsWith('a'))!.heldoutComposite).toBe('—')
  })
})

import { vi } from 'vitest'
import { authHeaders } from './api'

describe('authHeaders', () => {
  it('sends X-API-Key only when a key is configured', () => {
    vi.stubEnv('VITE_API_KEY', '')
    expect(authHeaders()).toEqual({})
    vi.stubEnv('VITE_API_KEY', 'k-123')
    expect(authHeaders()).toEqual({ 'X-API-Key': 'k-123' })
    vi.unstubAllEnvs()
  })
})
