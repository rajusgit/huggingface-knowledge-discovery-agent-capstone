import { benchmarkRows, type BenchmarkRecord } from '../api'

export default function Benchmarks({ records }: { records: BenchmarkRecord[] }) {
  if (!records.length) return null
  const cols = records.map((r) => ({ r, rows: benchmarkRows(r) }))
  const labels = [...new Set(cols.flatMap((c) => c.rows.map((x) => x.label)))]
  return (
    <section className="mt-3" aria-label="Benchmark results">
      <div className="mb-1 flex items-center gap-2 text-sm">
        <span className="rounded bg-green-100 px-2 py-0.5 text-xs font-semibold text-green-800">Measured here</span>
        <span className="text-slate-600">
          Run in our sandbox on the dataset shown. These are not publisher-reported scores and are not comparable to model-card numbers.
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr>
              <th className="border bg-slate-50 p-2 text-left"> </th>
              {cols.map(({ r }) => <th key={r.job_id} className="border bg-slate-50 p-2 text-left">{r.model_id}</th>)}
            </tr>
          </thead>
          <tbody>
            {labels.map((label) => (
              <tr key={label}>
                <th scope="row" className="border bg-slate-50 p-2 text-left font-medium">{label}</th>
                {cols.map(({ r, rows }) => (
                  <td key={r.job_id} className="border p-2">{rows.find((x) => x.label === label)?.value ?? '—'}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
