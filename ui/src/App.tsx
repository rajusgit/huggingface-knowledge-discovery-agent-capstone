import { useState } from 'react'
import Chat from './components/Chat'
import Compare from './components/Compare'
import Discover from './components/Discover'

export default function App() {
  const [tab, setTab] = useState<'chat' | 'discover' | 'compare'>('chat')
  const btn = (t: typeof tab) =>
    `rounded px-3 py-1 text-sm ${tab === t ? 'bg-blue-700 text-white' : 'bg-white text-slate-700 border border-slate-300'}`
  return (
    <div>
      <header className="mx-auto flex max-w-3xl items-center gap-2 px-4 pt-4">
        <h1 className="mr-auto text-xl font-semibold">Hugging Face Knowledge Agent</h1>
        <button className={btn('chat')} onClick={() => setTab('chat')}>Chat</button>
        <button className={btn('discover')} onClick={() => setTab('discover')}>Discover</button>
        <button className={btn('compare')} onClick={() => setTab('compare')}>Compare</button>
      </header>
      {tab === 'chat' ? <Chat /> : tab === 'discover' ? <Discover /> : <Compare />}
    </div>
  )
}
