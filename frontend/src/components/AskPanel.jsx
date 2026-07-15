import React, { useState } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import { Send, Loader2, Sparkles, FileText, HelpCircle } from 'lucide-react'
import { useToast } from '../contexts/ToastContext'

const SCORE_TOOLTIPS = {
  Faithfulness: 'How well the answer sticks to facts from your actual documents (vs. making things up).',
  Coverage: 'How thoroughly the answer addresses your full question.',
}

export const ScoreBadge = ({ label, score }) => {
  let color = 'bg-red-500/15 text-red-400 border-red-500/20';
  if (score >= 0.7) color = 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20';
  else if (score >= 0.4) color = 'bg-yellow-500/15 text-yellow-400 border-yellow-500/20';

  const tooltip = SCORE_TOOLTIPS[label]

  return (
    <div className="relative group">
      <div className={`px-3 py-1 rounded-full border flex items-center gap-2 ${color} text-sm font-medium cursor-help`}>
        <span className="opacity-75">{label}</span>
        <span>{score.toFixed(2)}</span>
        {tooltip && <HelpCircle className="w-3 h-3 opacity-50" />}
      </div>
      {tooltip && (
        <div className="absolute bottom-full left-0 mb-2 px-3 py-2 bg-slate-900 border border-white/10 rounded-lg text-xs text-slate-300 w-64 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none shadow-xl z-20">
          {tooltip}
        </div>
      )}
    </div>
  );
};

const EXAMPLE_QUESTIONS = [
  'What are the main conclusions?',
  'Summarize the key findings',
  'What methodology was used?',
  'What are the limitations?',
]

export default function AskPanel() {
  const [query, setQuery] = useState('')
  const [answerData, setAnswerData] = useState(null)
  const [loading, setLoading] = useState(false)
  const toast = useToast()

  const handleAsk = async (searchQuery) => {
    const q = searchQuery || query
    if (!q.trim()) return

    setLoading(true)
    setAnswerData(null)
    try {
      const res = await axios.post('/api/qa', { query: q, top_k: 5 })
      setAnswerData(res.data)
    } catch (err) {
      const detail = err.response?.data?.detail || err.message
      toast.error(`Failed to get answer: ${detail}`)
    }
    setLoading(false)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    handleAsk()
  }

  const handleExampleClick = (q) => {
    setQuery(q)
    handleAsk(q)
  }

  return (
    <div className="space-y-8 flex flex-col h-full">
      <div>
        <h2 className="text-xl font-semibold text-white">Ask AI</h2>
        <p className="text-sm text-slate-500 mt-1">Ask questions in plain English and get AI-generated answers with citations from your documents.</p>
      </div>

      <form onSubmit={handleSubmit} className="flex gap-3 shrink-0">
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
          <h3 className="text-slate-300 font-medium mb-2 text-lg">Ask anything about your documents</h3>
          <p className="text-slate-500 text-sm max-w-md mb-6">
            The AI will find relevant passages and generate an answer with quality scores showing how reliable it is.
          </p>
          <div className="flex flex-wrap gap-2 justify-center">
            {EXAMPLE_QUESTIONS.map((q, i) => (
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
          <span className="mt-4 text-sm text-slate-400 font-medium tracking-wide">Finding answers in your documents...</span>
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

          {/* Source Citations */}
          {answerData.sources && answerData.sources.length > 0 && (
            <div className="mt-8 pt-6 border-t border-white/5">
              <h4 className="text-xs uppercase tracking-widest text-slate-400 font-semibold mb-4">Sources Used ({answerData.sources.length})</h4>
              <div className="grid gap-3">
                {answerData.sources.map((src, i) => (
                  <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-slate-800/30 border border-white/5">
                    <div className="w-8 h-8 rounded-lg bg-slate-700/50 flex items-center justify-center text-slate-400 flex-shrink-0">
                      <FileText className="w-4 h-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-slate-300 font-medium truncate">{src.source_file}</div>
                      <div className="text-xs text-slate-500">Section #{src.chunk_index}</div>
                    </div>
                    <div className="text-xs font-mono text-emerald-400 flex-shrink-0">
                      {(src.score * 100).toFixed(0)}% relevant
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
