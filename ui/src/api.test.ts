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
