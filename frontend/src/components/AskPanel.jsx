import React, { useState } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import { Send, Loader2, Sparkles } from 'lucide-react'

export const ScoreBadge = ({ label, score }) => {
  let color = 'bg-red-500/15 text-red-400 border-red-500/20';
  if (score >= 0.7) color = 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20';
  else if (score >= 0.4) color = 'bg-yellow-500/15 text-yellow-400 border-yellow-500/20';

  return (
    <div className={`px-3 py-1 rounded-full border flex items-center gap-2 ${color} text-sm font-medium`}>
      <span className="opacity-75">{label}</span>
      <span>{score.toFixed(2)}</span>
    </div>
  );
};

export default function AskPanel() {
  const [query, setQuery] = useState('')
  const [answerData, setAnswerData] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleAsk = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setAnswerData(null)
    try {
      const res = await axios.post('/api/qa', { query, top_k: 5 })
      setAnswerData(res.data)
    } catch (err) {
      console.error(err)
    }
    setLoading(false)
  }

  return (
    <div className="space-y-8 flex flex-col h-full">
      <h2 className="text-xl font-semibold text-white">Generative Q&A</h2>

      <form onSubmit={handleAsk} className="flex gap-3 shrink-0">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about your documents..."
          className="flex-1 bg-[#1e2535] border border-white/5 text-slate-200 rounded-xl px-5 py-3.5 focus:outline-none focus:border-emerald-500/50 focus:ring-0 transition-colors duration-150 placeholder:text-slate-600 shadow-inner"
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="bg-emerald-500 hover:bg-emerald-400 disabled:opacity-60 disabled:hover:bg-emerald-500 text-black px-6 py-3.5 rounded-xl font-medium transition-colors duration-150 flex items-center justify-center min-w-[120px] gap-2 shadow-lg shadow-emerald-500/20"
        >
          {loading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <>
              <Send className="w-5 h-5" />
              <span>Ask</span>
            </>
          )}
        </button>
      </form>

      {!answerData && !loading && (
        <div className="flex-1 flex flex-col items-center justify-center text-center p-12 glass-card">
          <div className="w-16 h-16 rounded-full bg-slate-800/50 flex items-center justify-center text-slate-500 mb-6">
            <Sparkles className="w-8 h-8" />
          </div>
          <h3 className="text-slate-300 font-medium mb-2 text-lg">Ask anything</h3>
          <p className="text-slate-500 text-sm max-w-md">
            Query your indexed documents using natural language. The AI will retrieve context and generate an exact answer with citations.
          </p>
        </div>
      )}

      {loading && (
        <div className="flex-1 flex flex-col items-center justify-center glass-card">
          <div className="flex items-center gap-2 text-emerald-500">
            <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
            <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
            <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
          </div>
          <span className="mt-4 text-sm text-slate-400 font-medium tracking-wide">Synthesizing Answer...</span>
        </div>
      )}

      {answerData && !loading && (
        <div className="flex-1 overflow-y-auto glass-card p-8 shadow-xl">
          <div className="flex gap-3 mb-8 pb-6 border-b border-white/5">
            <ScoreBadge label="Faithfulness" score={answerData.faithfulness_score} />
            <ScoreBadge label="Coverage" score={answerData.coverage_score} />
          </div>
          
          <div className="markdown-body">
            <ReactMarkdown>{answerData.answer}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  )
}
