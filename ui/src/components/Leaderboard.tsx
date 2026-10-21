import { useEffect, useState } from 'react'
import { fetchLeaderboard, leaderboardRows, type LeaderboardData } from '../api'

export default function Leaderboard() {
  const [data, setData] = useState<LeaderboardData | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    const ctl = new AbortController()
    fetchLeaderboard(ctl.signal).then(setData).catch((e) => !ctl.signal.aborted && setError(e instanceof Error ? e.message : 'Failed to load'))
    return () => ctl.abort()
  }, [])
  if (error) return <p role="alert" className="mx-auto max-w-4xl px-4 py-4 text-red-700">{error}</p>
  if (!data) return <p className="mx-auto max-w-4xl px-4 py-4 text-slate-500">Loading…</p>
  if (!data.variants.length) return <p className="mx-auto max-w-4xl px-4 py-4 text-slate-500">{data.message ?? 'No results yet.'}</p>
  const rows = leaderboardRows(data)
  const th = 'border bg-slate-50 p-2 text-left'
  return (
    <div className="mx-auto max-w-5xl px-4 py-4">
      <p className="mb-2 text-sm text-slate-600">
        Experiment <code>{data.experiment_id}</code>. Chunking × retrieval × reranking; Recall@5 shows mean [95% CI]. The winner is chosen on the held-out split.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr>
              {['Variant', 'Dev score', 'Held-out score', 'Recall@5 (dev)', 'Retrieval p50 ms', 'Retrieval p95 ms', 'End-to-end p95 ms', '$ / query'].map((h) => <th key={h} className={th}>{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.variant} className={r.isWinner ? 'bg-green-50 font-medium' : ''}>
                <td className="border p-2">{r.variant}{r.isWinner && <span className="ml-2 rounded bg-green-200 px-1 text-xs">winner</span>}</td>
                <td className="border p-2">{r.devComposite}</td>
                <td className="border p-2">{r.heldoutComposite}</td>
                <td className="border p-2">{r.recall}</td>
                <td className="border p-2">{r.p50}</td>
                <td className="border p-2">{r.p95}</td>
                <td className="border p-2">{r.e2eP95}</td>
                <td className="border p-2">{r.cost}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
