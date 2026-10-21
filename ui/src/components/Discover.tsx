import { useState } from 'react'
import { searchHub, type HubKind, type HubResponse, type SearchFilters } from '../api'
import ResultCard from './ResultCard'

const SORTS = ['downloads', 'likes', 'lastModified', 'createdAt', 'trendingScore']

export default function Discover() {
  const [kind, setKind] = useState<HubKind>('models')
  const [f, setF] = useState<SearchFilters>({ sort: 'downloads', limit: 10 })
  const [data, setData] = useState<HubResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const set = (k: keyof SearchFilters) => (e: { target: { value: string } }) =>
    setF((cur) => ({ ...cur, [k]: k === 'limit' ? Number(e.target.value) : e.target.value }))

  async function submit() {
    setLoading(true)
    setError(null)
    try {
      setData(await searchHub(kind, f))
    } catch (err) {
      setData(null)
      setError(err instanceof Error ? err.message : 'Search failed')
    }
    setLoading(false)
  }

  const input = 'rounded border border-slate-300 px-2 py-1 text-sm'
  return (
    <div className="mx-auto max-w-3xl px-4 py-4">
      <form className="grid grid-cols-2 gap-2 sm:grid-cols-4" onSubmit={(e) => { e.preventDefault(); void submit() }}>
        <select className={input} aria-label="Type" value={kind} onChange={(e) => setKind(e.target.value as HubKind)}>
          <option value="models">Models</option>
          <option value="datasets">Datasets</option>
        </select>
        <input className={`${input} col-span-1 sm:col-span-3`} aria-label="Search" placeholder="Search…" value={f.q ?? ''} onChange={set('q')} />
        {kind === 'models' && (
          <>
            <input className={input} aria-label="Task" placeholder="Task (e.g. sentence-similarity)" value={f.task ?? ''} onChange={set('task')} />
            <input className={input} aria-label="Library" placeholder="Library" value={f.library ?? ''} onChange={set('library')} />
          </>
        )}
        <input className={input} aria-label="Language" placeholder="Language (e.g. fr)" value={f.language ?? ''} onChange={set('language')} />
        <input className={input} aria-label="License" placeholder="License (e.g. mit)" value={f.license ?? ''} onChange={set('license')} />
        <select className={input} aria-label="Sort" value={f.sort} onChange={set('sort')}>
          {SORTS.map((s) => <option key={s}>{s}</option>)}
        </select>
        <input className={input} aria-label="Limit" type="number" min={1} max={50} value={f.limit} onChange={set('limit')} />
        <button className="rounded bg-blue-700 px-4 py-1 text-white disabled:opacity-50 sm:col-span-4" disabled={loading}>
          {loading ? 'Searching…' : 'Search'}
        </button>
      </form>
      {error && <p role="alert" className="mt-3 text-red-700">{error}</p>}
      {data && data.results.length === 0 && <p className="mt-3 text-slate-500">No results for these filters.</p>}
      {data && data.results.length > 0 && (
        <ul className="mt-3 space-y-2">
          {data.results.map((r) => <ResultCard key={r.id} r={r} fetchedAt={data.fetched_at} />)}
        </ul>
      )}
    </div>
  )
}
