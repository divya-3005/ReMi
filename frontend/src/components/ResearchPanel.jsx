import React, { useState, useEffect } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import EvalDashboard from './EvalDashboard'
import { motion } from 'framer-motion'
import { BrainCircuit, Loader2, CheckCircle2, XCircle, Clock } from 'lucide-react'
import { ScoreBadge } from './AskPanel'
import { useToast } from '../contexts/ToastContext'

const EXAMPLE_QUESTIONS = [
  'What is the comprehensive overview of the main topic?',
  'Compare and contrast the different approaches mentioned.',
  'What is the detailed timeline of events?',
]

export default function ResearchPanel() {
  const [question, setQuestion] = useState('')
  const [reportId, setReportId] = useState(null)
  const [reportData, setReportData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const toast = useToast()

  // Polling effect
  useEffect(() => {
    let interval;
    if (reportId && (!reportData || reportData.status === 'running')) {
      interval = setInterval(async () => {
        try {
          const res = await axios.get(`/api/research/reports/${reportId}`)
          setReportData(res.data)
        } catch (err) {
          console.error(err)
        }
      }, 3000)
    }
    return () => clearInterval(interval)
  }, [reportId, reportData])

  // Timer effect
  useEffect(() => {
    let timer;
    if (loading) {
      timer = setInterval(() => {
        setElapsed(prev => prev + 1)
      }, 1000)
    } else {
      setElapsed(0)
    }
    return () => clearInterval(timer)
  }, [loading])

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60)
    const s = seconds % 60
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  const handleResearch = async (searchQuestion) => {
    const q = searchQuestion || question
    if (!q.trim()) return

    setLoading(true)
    setReportId(null)
    setReportData(null)
    setElapsed(0)
    try {
      const res = await axios.post('/api/research', { question: q, min_confidence: 0.3 })
      setReportId(res.data.report_id)
    } catch (err) {
      const detail = err.response?.data?.detail || err.message
      toast.error(`Failed to start research: ${detail}`)
      setLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    handleResearch()
  }

  const handleExampleClick = (q) => {
    setQuestion(q)
    handleResearch(q)
  }

  // Determine if we're done generating
  useEffect(() => {
    if (loading && reportData && (reportData.status === 'complete' || reportData.status === 'failed')) {
      setLoading(false)
      if (reportData.status === 'failed') {
        toast.error('Research task failed. Please try again.')
      } else {
        toast.success('Research report completed!')
      }
    }
  }, [reportData, toast, loading])

  return (
    <div className="space-y-8 flex flex-col h-full">
      <div>
        <h2 className="text-xl font-semibold text-white">Deep Research</h2>
        <p className="text-sm text-slate-500 mt-1">Ask a complex question and our AI will research it thoroughly, piece by piece, and write a cited report.</p>
      </div>

      <form onSubmit={handleSubmit} className="flex gap-3 shrink-0">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Enter a complex research question..."
          className="flex-1 bg-[#1e2535] border border-white/5 text-slate-200 rounded-xl px-5 py-3.5 focus:outline-none focus:border-emerald-500/50 focus:ring-0 transition-colors duration-150 placeholder:text-slate-600 shadow-inner"
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="bg-emerald-500 hover:bg-emerald-400 disabled:opacity-60 disabled:hover:bg-emerald-500 text-black px-6 py-3.5 rounded-xl font-medium transition-colors duration-150 flex items-center justify-center min-w-[150px] gap-2 shadow-lg shadow-emerald-500/20"
        >
          {loading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <>
              <BrainCircuit className="w-5 h-5" />
              <span>Run Research</span>
            </>
          )}
        </button>
      </form>

      {!reportData && !loading && (
        <div className="flex-1 flex flex-col items-center justify-center text-center p-12 glass-card">
          <div className="w-16 h-16 rounded-full bg-slate-800/50 flex items-center justify-center text-slate-500 mb-6">
            <BrainCircuit className="w-8 h-8" />
          </div>
          <h3 className="text-slate-300 font-medium mb-2 text-lg">Multi-Agent Workflow</h3>
          <p className="text-slate-500 text-sm max-w-md mb-6">
            Unlike simple Q&A, Deep Research breaks your question down into smaller parts, researches each part separately across all your documents, and synthesizes a comprehensive final report.
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

      {reportData && (
        <div className="flex-1 flex flex-col gap-6 overflow-hidden">
          
          {/* Sub-questions Timeline */}
          <div className="glass-card p-6 shrink-0 shadow-lg">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xs uppercase tracking-widest text-slate-400 font-semibold">Research Progress</h3>
              <div className="flex items-center gap-4">
                {loading && (
                  <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
                    <Clock className="w-3.5 h-3.5" />
                    <span>{formatTime(elapsed)}</span>
                    <span className="text-slate-600">/ ~1:00</span>
                  </div>
                )}
                <div className="flex items-center gap-2 text-xs font-mono font-medium">
                  <span className="text-slate-500">Status:</span>
                  <span className={`px-2 py-0.5 rounded ${
                    reportData.status === 'running' ? 'bg-amber-500/10 text-amber-400' : 
                    reportData.status === 'complete' ? 'bg-emerald-500/10 text-emerald-400' : 
                    'bg-red-500/10 text-red-400'
                  }`}>
                    {reportData.status.toUpperCase()}
                  </span>
                </div>
              </div>
            </div>
            
            {reportData.sub_questions && reportData.sub_questions.length > 0 ? (
              <div className="relative pl-3">
                <div className="absolute left-[15px] top-3 bottom-3 w-0.5 bg-slate-800"></div>
                <div className="space-y-6">
                  {reportData.sub_questions.map((sq, i) => (
                    <motion.div 
                      initial={{ opacity: 0, x: -10 }} 
                      animate={{ opacity: 1, x: 0 }}
                      key={i} 
                      className="flex items-start gap-4 relative"
                    >
                      <div className="relative z-10 flex-shrink-0 mt-0.5 bg-[#161b27]">
                        {sq.status === 'pending' && (
                          <div className="w-6 h-6 rounded-full border-2 border-slate-700 flex items-center justify-center bg-[#161b27]"></div>
                        )}
                        {sq.status === 'running' && (
                          <div className="w-6 h-6 rounded-full flex items-center justify-center bg-[#161b27]">
                            <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-ping absolute"></span>
                            <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full relative"></span>
                          </div>
                        )}
                        {sq.status === 'done' && (
                          <div className="w-6 h-6 bg-[#161b27] rounded-full flex items-center justify-center">
                            <CheckCircle2 className="w-6 h-6 text-emerald-500" />
                          </div>
                        )}
                        {sq.status === 'failed' && (
                          <div className="w-6 h-6 bg-[#161b27] rounded-full flex items-center justify-center">
                            <XCircle className="w-6 h-6 text-red-500" />
                          </div>
                        )}
                      </div>
                      <span className={`text-sm pt-0.5 font-medium ${
                        sq.status === 'pending' ? 'text-slate-500' : 
                        sq.status === 'running' ? 'text-white' :
                        sq.status === 'done' ? 'text-slate-300' : 
                        'text-red-400 line-through'
                      }`}>
                        {sq.question}
                      </span>
                    </motion.div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-3 text-slate-500 text-sm italic py-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Planning research strategy...
              </div>
            )}
          </div>

          {/* Final Report */}
          {reportData.final_report && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex-1 overflow-y-auto glass-card p-8 shadow-xl relative"
            >
              <div className="flex gap-3 mb-8 pb-6 border-b border-white/5 sticky top-0 bg-[#161b27]/90 backdrop-blur-md z-10 pt-2 -mt-2">
                <ScoreBadge label="Faithfulness" score={reportData.faithfulness_score} />
                <ScoreBadge label="Coverage" score={reportData.coverage_score} />
              </div>
              
              <div className="markdown-body">
                <ReactMarkdown>{reportData.final_report}</ReactMarkdown>
              </div>
              
              {reportData.status === 'complete' && reportData.eval_result && (
                <EvalDashboard evalResult={reportData.eval_result} />
              )}
            </motion.div>
          )}
          
        </div>
      )}
    </div>
  )
}
