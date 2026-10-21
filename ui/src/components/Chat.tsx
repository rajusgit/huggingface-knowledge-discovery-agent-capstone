import { useRef, useState } from 'react'
import { resumeChat, safeUrl, splitCitations, streamChat, threadIdFor, type Approval, type BenchmarkRecord, type ChatEvent, type Citation, type LiveSource } from '../api'
import Benchmarks from './Benchmarks'
import Citations from './Citations'

interface Turn {
  question: string
  answer: string
  citations: Citation[]
  live: LiveSource[]
  bench: BenchmarkRecord[]
  approval?: Approval & { threadId: string }
  status?: string
  error?: string
  done: boolean
}

export default function Chat() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [active, setActive] = useState<number | null>(null)
  const [memory, setMemory] = useState(false)
  const stableThread = useRef(crypto.randomUUID())
  const last = turns[turns.length - 1]
  const busy = turns.length > 0 && !last.done && !last.approval

  const patchLast = (f: (t: Turn) => Turn) =>
    setTurns((ts) => ts.map((t, i) => (i === ts.length - 1 ? f(t) : t)))

  async function send() {
    const question = input.trim()
    if (!question || busy) return
    setInput('')
    setActive(null)
    setTurns((ts) => [...ts, { question, answer: '', citations: [], live: [], bench: [], done: false }])
    await consume(streamChat(question, threadIdFor(memory, stableThread.current), memory), stableThread.current)
  }

  async function consume(events: AsyncGenerator<ChatEvent>, thread: string) {
    try {
      for await (const e of events) {
        if (e.type === 'token') patchLast((t) => ({ ...t, answer: t.answer + e.text }))
        else if (e.type === 'benchmark_status') patchLast((t) => ({ ...t, status: e.jobs.map((j) => `${j.model_id}: ${j.status}`).join(' · ') }))
        else if (e.type === 'approval_required') patchLast((t) => ({ ...t, approval: { ...e, threadId: thread } }))
        else if (e.type === 'final') patchLast((t) => ({ ...t, citations: e.citations, live: e.live_sources, bench: e.benchmarks, status: undefined, done: true }))
        else if (e.type === 'error') patchLast((t) => ({ ...t, error: e.message, done: true }))
      }
    } catch (err) {
      patchLast((t) => ({ ...t, error: err instanceof Error ? err.message : 'Request failed', done: true }))
    }
    patchLast((t) => (t.approval ? { ...t, done: false } : { ...t, done: true }))
  }

  async function decide(approved: boolean) {
    const t = turns[turns.length - 1]
    if (!t?.approval) return
    const thread = t.approval.threadId
    patchLast((x) => ({ ...x, approval: undefined, done: false, status: approved ? 'Starting benchmark…' : undefined }))
    await consume(resumeChat(thread, approved, memory), thread)
  }

  return (
    <div className="mx-auto flex h-[calc(100vh-3.5rem)] max-w-3xl flex-col px-4 py-2">
      <label className="mb-2 flex items-center gap-2 text-sm text-slate-600">
        <input type="checkbox" checked={memory} onChange={(e) => setMemory(e.target.checked)} />
        Remember this conversation (stored on the server; off by default)
      </label>
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
            {t.status && <p className="mt-2 text-sm text-slate-600" role="status">{t.status}</p>}
            {t.approval && (
              <div className="mt-3 rounded border border-amber-300 bg-amber-50 p-3 text-sm" role="alertdialog" aria-label="Approve benchmark">
                <p className="font-medium">Approve running a benchmark?</p>
                <ul className="mt-1 list-disc pl-5">
                  <li>Models: {t.approval.models.join(', ')}</li>
                  <li>Dataset: {t.approval.dataset} (seed {t.approval.seed})</li>
                  <li>Limits: {t.approval.limits.max_runtime_seconds}s runtime, {t.approval.limits.max_memory_mb} MB memory, sandboxed, no remote code</li>
                </ul>
                <div className="mt-2 flex gap-2">
                  <button className="rounded bg-blue-700 px-3 py-1 text-white" onClick={() => void decide(true)}>Approve</button>
                  <button className="rounded border border-slate-300 bg-white px-3 py-1" onClick={() => void decide(false)}>Reject</button>
                </div>
              </div>
            )}
            <Benchmarks records={t.bench} />
            {t.error && <p role="alert" className="mt-2 text-red-700">{t.error}</p>}
            <Citations citations={t.citations} active={active} />
            {t.live.length > 0 && (
              <div className="mt-3 text-xs text-slate-600">
                <div className="font-medium">Live Hugging Face data</div>
                <ul className="list-disc pl-5">
                  {t.live.slice(0, 8).map((s, k) => {
                    const href = safeUrl(s.source_url)
                    return (
                      <li key={k}>
                        {href ? <a className="text-blue-700 underline" href={href} target="_blank" rel="noopener noreferrer">{s.source_url}</a> : s.source_url}
                        {s.fetched_at && <> · fetched {s.fetched_at}</>}
                      </li>
                    )
                  })}
                </ul>
              </div>
            )}
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
