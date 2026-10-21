import { safeUrl, type Citation } from '../api'

export default function Citations({ citations, active }: { citations: Citation[]; active: number | null }) {
  if (!citations.length) return null
  return (
    <ol className="mt-3 space-y-2 text-sm" aria-label="Sources">
      {citations.map((c) => {
        const href = safeUrl(c.source_url)
        return (
          <li
            key={c.chunk_id}
            id={`cite-${c.n}`}
            className={`rounded border p-2 ${active === c.n ? 'border-blue-500 bg-blue-50' : 'border-slate-200 bg-white'}`}
          >
            <div className="font-medium">
              [{c.n}] {c.heading_path.join(' › ')}
            </div>
            <div className="text-xs text-slate-500">
              {href ? (
                <a className="text-blue-700 underline" href={href} target="_blank" rel="noopener noreferrer">
                  {c.source_url}
                </a>
              ) : (
                c.source_url
              )}{' '}
              · rev {c.revision.slice(0, 8)} · score {c.score.toFixed(2)}
            </div>
            <details className="mt-1">
              <summary className="cursor-pointer text-xs text-slate-600">Retrieved text</summary>
              <p className="mt-1 whitespace-pre-wrap text-xs text-slate-700">{c.content}</p>
            </details>
          </li>
        )
      })}
    </ol>
  )
}
