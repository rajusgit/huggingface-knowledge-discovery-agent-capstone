import { useRef, useState } from 'react'
import { splitCitations, streamChat, type Citation } from '../api'
import Citations from './Citations'

interface Turn {
  question: string
  answer: string
  citations: Citation[]
  error?: string
  done: boolean
}

export default function Chat() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [active, setActive] = useState<number | null>(null)
  const threadId = useRef(crypto.randomUUID())
  const busy = turns.length > 0 && !turns[turns.length - 1].done

  const patchLast = (f: (t: Turn) => Turn) =>
    setTurns((ts) => ts.map((t, i) => (i === ts.length - 1 ? f(t) : t)))

  async function send() {
    const question = input.trim()
    if (!question || busy) return
    setInput('')
    setActive(null)
    setTurns((ts) => [...ts, { question, answer: '', citations: [], done: false }])
    try {
      for await (const e of streamChat(question, threadId.current)) {
        if (e.type === 'token') patchLast((t) => ({ ...t, answer: t.answer + e.text }))
        else if (e.type === 'final') patchLast((t) => ({ ...t, citations: e.citations, done: true }))
        else patchLast((t) => ({ ...t, error: e.message, done: true }))
      }
    } catch (err) {
      patchLast((t) => ({ ...t, error: err instanceof Error ? err.message : 'Request failed', done: true }))
    }
    patchLast((t) => ({ ...t, done: true }))
  }

  return (
    <div className="mx-auto flex h-screen max-w-3xl flex-col px-4 py-4">
      <h1 className="mb-3 text-xl font-semibold">Hugging Face Knowledge Agent</h1>
      <div className="flex-1 space-y-6 overflow-y-auto pb-4" aria-live="polite">
        {turns.length === 0 && (
          <p className="text-slate-500">Ask about Transformers, e.g. “How do I limit the length of generated text?”</p>
        )}
        {turns.map((t, i) => (
          <div key={i}>
            <div className="rounded bg-slate-100 p-2 font-medium">{t.question}</div>
            <p className="mt-2 whitespace-pre-wrap">
              {splitCitations(t.answer).map((p, j) =>
                'n' in p ? (
                  <a
                    key={j}
                    href={`#cite-${p.n}`}
                    onClick={() => setActive(p.n)}
                    className="mx-0.5 rounded bg-blue-100 px-1 text-xs font-semibold text-blue-800"
                  >
                    {p.n}
                  </a>
                ) : (
                  <span key={j}>{p.text}</span>
                ),
              )}
              {!t.done && <span className="animate-pulse">▍</span>}
            </p>
            {t.error && <p role="alert" className="mt-2 text-red-700">{t.error}</p>}
            <Citations citations={t.citations} active={active} />
          </div>
        ))}
      </div>
      <form
        className="flex gap-2 border-t pt-3"
        onSubmit={(e) => {
          e.preventDefault()
          void send()
        }}
      >
        <input
          className="flex-1 rounded border border-slate-300 px-3 py-2"
          placeholder="Ask a question…"
          aria-label="Question"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button className="rounded bg-blue-700 px-4 py-2 text-white disabled:opacity-50" disabled={busy || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}
