import { useState } from 'react'
import { buildConstraints, compareResources, parseIds, safeUrl, type CompareResponse, type CompareRow, type ConstraintForm, type HubKind } from '../api'

const ROWS: { key: keyof CompareRow; label: string; sensitive?: boolean }[] = [
  { key: 'pipeline_tag', label: 'Task' },
  { key: 'library_name', label: 'Library' },
  { key: 'license', label: 'License' },
  { key: 'parameters', label: 'Parameters' },
  { key: 'languages', label: 'Languages' },
  { key: 'downloads', label: 'Downloads', sensitive: true },
  { key: 'likes', label: 'Likes', sensitive: true },
  { key: 'revision', label: 'Revision' },
  { key: 'gated', label: 'Gated' },
]

function show(v: CompareRow[keyof CompareRow]): string | null {
  if (v === null || v === undefined || (Array.isArray(v) && v.length === 0)) return null
  if (Array.isArray(v)) return v.length > 6 ? `${v.slice(0, 6).join(', ')} … (+${v.length - 6})` : v.join(', ')
  if (typeof v === 'number') return v.toLocaleString()
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  return String(v).length > 12 && /^[0-9a-f]{40}$/.test(String(v)) ? String(v).slice(0, 8) : String(v)
}

export default function Compare() {
  const [kind, setKind] = useState<HubKind>('models')
  const [ids, setIds] = useState('')
  const [form, setForm] = useState<ConstraintForm>({ licenses: '', maxParamsM: '', language: '', task: '' })
  const [data, setData] = useState<CompareResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function submit() {
    setLoading(true)
    setError(null)
    try {
      setData(await compareResources(parseIds(ids), kind, buildConstraints(form)))
    } catch (err) {
      setData(null)
      setError(err instanceof Error ? err.message : 'Comparison failed')
    }
    setLoading(false)
  }

  const input = 'rounded border border-slate-300 px-2 py-1 text-sm'
  const field = (k: keyof ConstraintForm) => (e: { target: { value: string } }) => setForm((f) => ({ ...f, [k]: e.target.value }))
  return (
    <div className="mx-auto max-w-4xl px-4 py-4">
      <form className="grid gap-2 sm:grid-cols-4" onSubmit={(e) => { e.preventDefault(); void submit() }}>
        <select className={input} aria-label="Type" value={kind} onChange={(e) => setKind(e.target.value as HubKind)}>
          <option value="models">Models</option>
          <option value="datasets">Datasets</option>
        </select>
        <textarea className={`${input} sm:col-span-3`} rows={2} aria-label="Ids" placeholder="2-5 ids, e.g. BAAI/bge-small-en-v1.5, intfloat/multilingual-e5-small" value={ids} onChange={(e) => setIds(e.target.value)} />
        <input className={input} aria-label="Allowed licenses" placeholder="Allowed licenses (mit, apache-2.0)" value={form.licenses} onChange={field('licenses')} />
        <input className={input} aria-label="Max parameters (millions)" placeholder="Max params (millions)" value={form.maxParamsM} onChange={field('maxParamsM')} />
        <input className={input} aria-label="Language" placeholder="Language (fr)" value={form.language} onChange={field('language')} />
        <input className={input} aria-label="Task" placeholder="Task" value={form.task} onChange={field('task')} />
        <button className="rounded bg-blue-700 px-4 py-1 text-white disabled:opacity-50 sm:col-span-4" disabled={loading}>
          {loading ? 'Comparing…' : 'Compare'}
        </button>
      </form>
      {error && <p role="alert" className="mt-3 text-red-700">{error}</p>}
      {data && (
        <div className="mt-4">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr>
                  <th className="border bg-slate-50 p-2 text-left"> </th>
                  {data.models.map((m) => {
                    const href = safeUrl(m.source_url)
                    const chk = data.constraints_check?.[m.id]
                    return (
                      <th key={m.id} className="border bg-slate-50 p-2 text-left align-top">
                        {href ? <a className="text-blue-700 underline" href={href} target="_blank" rel="noopener noreferrer">{m.id}</a> : m.id}
                        {chk && (
                          <div className={`mt-1 text-xs font-normal ${chk.ok ? 'text-green-700' : 'text-red-700'}`}>
                            {chk.ok ? 'Meets constraints' : 'Violates constraints'}
                          </div>
                        )}
                      </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody>
                {ROWS.map((r) => (
                  <tr key={r.key}>
                    <th scope="row" className="border bg-slate-50 p-2 text-left font-medium">
                      {r.label}{r.sensitive && <span className="ml-1 rounded bg-slate-100 px-1 text-xs font-normal">time-sensitive</span>}
                    </th>
                    {data.models.map((m) => {
                      const v = show(m[r.key])
                      return (
                        <td key={m.id} className="border p-2" title={`Source: ${m.source_url} · fetched ${m.fetched_at}`}>
                          {v ?? <span className="italic text-amber-700">not stated</span>}
                        </td>
                      )
                    })}
                  </tr>
                ))}
                <tr>
                  <th scope="row" className="border bg-slate-50 p-2 text-left font-medium">Source · fetched</th>
                  {data.models.map((m) => (
                    <td key={m.id} className="border p-2 text-xs text-slate-500">{m.source_url.replace('https://huggingface.co/', 'hf.co/')}<br />{m.fetched_at}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          {data.constraints_check && (
            <ul className="mt-3 space-y-1 text-sm">
              {data.models.flatMap((m) => {
                const chk = data.constraints_check![m.id]
                return [
                  ...chk.violations.map((v) => <li key={`${m.id}-v-${v}`} className="text-red-700">✗ {m.id}: {v}</li>),
                  ...chk.unknown.map((u) => <li key={`${m.id}-u-${u}`} className="text-amber-700">? {m.id}: cannot check {u} (not stated)</li>),
                ]
              })}
            </ul>
          )}
          {data.warnings.length > 0 && (
            <ul className="mt-3 list-disc pl-5 text-sm text-amber-800" aria-label="Missing fields">
              {data.warnings.map((w) => <li key={`${w.id}-${w.field}`}>{w.message}</li>)}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
