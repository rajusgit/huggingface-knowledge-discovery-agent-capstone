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
