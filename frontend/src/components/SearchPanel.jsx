import React, { useState } from 'react'
import axios from 'axios'
import { motion } from 'framer-motion'
import { Search, Loader2, FileSearch } from 'lucide-react'
import { useToast } from '../contexts/ToastContext'

const EXAMPLE_QUERIES = [
  'Key findings and conclusions',
  'Methodology and approach',
  'Main challenges discussed',
  'Recommendations made',
]

export default function SearchPanel() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)
  const toast = useToast()

  const handleSearch = async (e) => {
    if (e) e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setSearched(true)
    try {
      const res = await axios.post('/api/search', { query, top_k: 5 })
      setResults(res.data.results)
      if (res.data.results.length === 0) {
        toast.info('No matching results found. Try a different query or upload more documents.')
      }
    } catch (err) {
      const detail = err.response?.data?.detail || err.message
      toast.error(`Search failed: ${detail}`)
      setResults([])
    }
    setLoading(false)
  }

  const handleExampleClick = (q) => {
    setQuery(q)
    // Auto-search after setting query
    setTimeout(() => {
      setQuery(q)
      setLoading(true)
      setSearched(true)
      axios.post('/api/search', { query: q, top_k: 5 })
        .then(res => {
          setResults(res.data.results)
          if (res.data.results.length === 0) {
            toast.info('No matching results found.')
          }
        })
        .catch(err => {
          const detail = err.response?.data?.detail || err.message
          toast.error(`Search failed: ${detail}`)
          setResults([])
        })
        .finally(() => setLoading(false))
    }, 0)
  }

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.05 }
    }
  }

  const item = {
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0 }
  }

  return (
    <div className="space-y-8 flex flex-col h-full">
      <div>
        <h2 className="text-xl font-semibold text-white">Search Your Documents</h2>
        <p className="text-sm text-slate-500 mt-1">Find relevant passages across all your uploaded files using natural language.</p>
      </div>

      <form onSubmit={handleSearch} className="flex gap-3 shrink-0">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Describe what you're looking for..."
          className="flex-1 bg-[#1e2535] border border-white/5 text-slate-200 rounded-xl px-5 py-3.5 focus:outline-none focus:border-emerald-500/50 focus:ring-0 transition-colors duration-150 placeholder:text-slate-600 shadow-inner"
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="bg-emerald-500 hover:bg-emerald-400 disabled:opacity-60 disabled:hover:bg-emerald-500 text-black px-6 py-3.5 rounded-xl font-medium transition-colors duration-150 flex items-center justify-center min-w-[140px] gap-2 shadow-lg shadow-emerald-500/20"
        >
          {loading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <>
              <Search className="w-5 h-5" />
              <span>Search</span>
            </>
          )}
        </button>
      </form>

      {!searched && !loading && (
        <div className="flex-1 flex flex-col items-center justify-center text-center p-12 glass-card">
          <div className="w-16 h-16 rounded-full bg-slate-800/50 flex items-center justify-center text-slate-500 mb-6">
            <FileSearch className="w-8 h-8" />
          </div>
          <h3 className="text-slate-300 font-medium mb-2 text-lg">Search across all your documents</h3>
          <p className="text-slate-500 text-sm max-w-md mb-6">
            Just describe what you're looking for in plain English. The AI understands meaning, not just exact keywords.
          </p>
          <div className="flex flex-wrap gap-2 justify-center">
            {EXAMPLE_QUERIES.map((q, i) => (
              <button
                key={i}
                onClick={() => handleExampleClick(q)}
                className="px-3.5 py-2 rounded-lg bg-slate-800/50 text-slate-400 text-xs font-medium hover:bg-slate-700/50 hover:text-slate-200 transition-colors border border-white/5"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {loading && (
        <div className="flex-1 flex flex-col items-center justify-center glass-card">
          <div className="flex items-center gap-2 text-emerald-500">
            <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
            <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
            <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
          </div>
          <span className="mt-4 text-sm text-slate-400 font-medium tracking-wide">Searching your documents...</span>
        </div>
      )}

      {searched && !loading && results.length === 0 && (
        <div className="flex-1 flex items-center justify-center glass-card p-8">
          <div className="text-slate-500 italic">No relevant passages found. Try rephrasing your query or upload more documents.</div>
        </div>
      )}

      {searched && !loading && results.length > 0 && (
        <div className="flex-1 overflow-y-auto pr-2 pb-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xs uppercase tracking-widest text-slate-400">Top Matches ({results.length})</h3>
          </div>
          <motion.div variants={container} initial="hidden" animate="show" className="grid gap-4">
            {results.map((res, i) => (
              <motion.div variants={item} key={i} className="glass-card overflow-hidden flex flex-col relative group">
                <div className="absolute left-0 top-0 bottom-0 w-1 bg-emerald-500/50 group-hover:bg-emerald-400 transition-colors"></div>
                <div className="p-6 pl-8">
                  <div className="flex justify-between items-center mb-4">
                    <div className="font-mono text-xs text-slate-400 font-medium">{res.source_file}</div>
                    <div className="text-xs font-mono font-medium text-slate-500">
                      Section #{res.chunk_index}
                    </div>
                  </div>
                  <p className="text-slate-300 text-sm leading-relaxed mb-6">{res.chunk_text}</p>
                  
                  {/* Progress bar for score */}
                  <div className="mt-auto">
                    <div className="flex justify-between items-center mb-1.5">
                      <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Relevance</span>
                      <span className="text-xs font-mono text-emerald-400 font-medium">{(res.score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-emerald-500/50 rounded-full" 
                        style={{ width: `${res.score * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      )}
    </div>
  )
}
