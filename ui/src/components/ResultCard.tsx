import { safeUrl, type HubResult } from '../api'

const fmt = (n: number | null) => (n === null ? '—' : n.toLocaleString())

export default function ResultCard({ r, fetchedAt }: { r: HubResult; fetchedAt: string }) {
  const href = safeUrl(r.source_url)
  return (
    <li className="rounded border border-slate-200 bg-white p-3">
      <div className="font-medium">
        {href ? (
          <a className="text-blue-700 underline" href={href} target="_blank" rel="noopener noreferrer">
            {r.id}
          </a>
        ) : (
          r.id
        )}
        {r.gated ? <span className="ml-2 rounded bg-amber-100 px-1 text-xs text-amber-800">gated</span> : null}
      </div>
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
        <span>License: {r.license ?? 'not stated'}</span>
        {r.pipeline_tag && <span>Task: {r.pipeline_tag}</span>}
        {r.library_name && <span>Library: {r.library_name}</span>}
      </div>
      <div className="mt-1 text-xs text-slate-600">
        <span title="Changes over time">⏱ Downloads: {fmt(r.downloads)} · Likes: {fmt(r.likes)}</span>{' '}
        <span className="rounded bg-slate-100 px-1">time-sensitive</span>
      </div>
      <div className="mt-1 text-xs text-slate-400">Fetched {fetchedAt}</div>
    </li>
  )
}
