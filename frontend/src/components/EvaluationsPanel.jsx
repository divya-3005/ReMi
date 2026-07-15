import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { Activity, RefreshCw, HelpCircle } from 'lucide-react';

const SCORE_TOOLTIPS = {
  Faithfulness: 'Did the AI stick to facts from your documents?',
  Relevance: 'Did the answer actually address your question?',
  Precision: 'Were the right parts of documents found?',
  Hallucination: 'Did the AI make anything up? (Lower is better)',
  Overall: 'Combined quality score of the answer'
}

export default function EvaluationsPanel() {
  const [evals, setEvals] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchEvals();
  }, []);

  const fetchEvals = async () => {
    setLoading(true);
    try {
      const [listRes, summaryRes] = await Promise.all([
        axios.get('/api/research/evals'),
        axios.get('/api/research/evals/summary')
      ]);
      const sorted = listRes.data.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      setEvals(sorted);
      setSummary(summaryRes.data);
    } catch (err) {
      console.error("Failed to fetch evals:", err);
    }
    setLoading(false);
  };

  const formatScore = (val) => {
    if (val === undefined || val === null) return '-';
    return (val * 100).toFixed(0) + '%';
  };

  const getColor = (val, reverse = false) => {
    if (val === undefined || val === null) return 'text-slate-500';
    if (reverse) {
      if (val <= 0.3) return 'text-emerald-400 drop-shadow-[0_0_8px_rgba(16,185,129,0.3)]';
      if (val <= 0.6) return 'text-yellow-400 drop-shadow-[0_0_8px_rgba(234,179,8,0.3)]';
      return 'text-red-400 drop-shadow-[0_0_8px_rgba(239,68,68,0.3)]';
    }
    if (val >= 0.7) return 'text-emerald-400 drop-shadow-[0_0_8px_rgba(16,185,129,0.3)]';
    if (val >= 0.4) return 'text-yellow-400 drop-shadow-[0_0_8px_rgba(234,179,8,0.3)]';
    return 'text-red-400 drop-shadow-[0_0_8px_rgba(239,68,68,0.3)]';
  };

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
    <div className="space-y-8 h-full flex flex-col">
      <div className="flex justify-between items-center shrink-0">
        <div>
          <h2 className="text-xl font-semibold text-white">Quality Scores</h2>
          <p className="text-sm text-slate-500 mt-1">Track the reliability and accuracy of AI-generated answers over time.</p>
        </div>
        <button 
          onClick={fetchEvals} 
          disabled={loading}
          className="bg-[#1e2535] hover:bg-slate-800 border border-white/5 text-slate-300 px-4 py-2 rounded-xl font-medium transition-colors duration-150 flex items-center gap-2"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {summary && summary.averages && (
        <div className="grid grid-cols-5 gap-4 shrink-0">
          {[
            { label: 'Avg Faithfulness', val: summary.averages.faithfulness, key: 'Faithfulness' },
            { label: 'Avg Relevance', val: summary.averages.answer_relevance, key: 'Relevance' },
            { label: 'Avg Precision', val: summary.averages.context_precision, key: 'Precision' },
            { label: 'Avg Hallucination', val: summary.averages.hallucination_risk, reverse: true, key: 'Hallucination' },
            { label: 'Overall Quality', val: summary.averages.overall_score, key: 'Overall' }
          ].map((s, i) => (
            <div key={i} className="glass-card p-5 text-center shadow-lg relative group">
              <div className="flex items-center justify-center gap-1.5 mb-2 cursor-help">
                <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">{s.label}</div>
                <HelpCircle className="w-3 h-3 text-slate-600 group-hover:text-slate-400 transition-colors" />
              </div>
              
              {/* Tooltip */}
              <div className="absolute -top-12 left-1/2 -translate-x-1/2 w-48 bg-slate-800 border border-white/10 rounded-lg p-2 text-xs text-slate-300 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 shadow-xl">
                {SCORE_TOOLTIPS[s.key]}
              </div>

              <div className={`text-3xl font-bold font-mono tracking-tighter ${getColor(s.val, s.reverse)}`}>
                {formatScore(s.val)}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="flex-1 glass-card overflow-hidden flex flex-col shadow-xl">
        <div className="overflow-x-auto flex-1">
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="text-[10px] uppercase tracking-widest bg-slate-900/50 text-slate-500 sticky top-0 backdrop-blur-md z-10">
              <tr>
                <th className="px-6 py-4 font-semibold">Date</th>
                <th className="px-6 py-4 font-semibold">Question</th>
                <th className="px-6 py-4 font-semibold group cursor-help relative">
                  <div className="flex items-center gap-1.5">
                    Faithfulness
                    <HelpCircle className="w-3 h-3 opacity-50" />
                  </div>
                  <div className="absolute hidden group-hover:block top-full mt-2 bg-slate-800 text-slate-300 text-xs p-2 rounded border border-white/10 w-48 normal-case font-normal shadow-xl">
                    {SCORE_TOOLTIPS['Faithfulness']}
                  </div>
                </th>
                <th className="px-6 py-4 font-semibold group cursor-help relative">
                  <div className="flex items-center gap-1.5">
                    Relevance
                    <HelpCircle className="w-3 h-3 opacity-50" />
                  </div>
                  <div className="absolute hidden group-hover:block top-full mt-2 bg-slate-800 text-slate-300 text-xs p-2 rounded border border-white/10 w-48 normal-case font-normal shadow-xl">
                    {SCORE_TOOLTIPS['Relevance']}
                  </div>
                </th>
                <th className="px-6 py-4 font-semibold group cursor-help relative">
                  <div className="flex items-center gap-1.5">
                    Precision
                    <HelpCircle className="w-3 h-3 opacity-50" />
                  </div>
                  <div className="absolute hidden group-hover:block top-full mt-2 bg-slate-800 text-slate-300 text-xs p-2 rounded border border-white/10 w-48 normal-case font-normal shadow-xl">
                    {SCORE_TOOLTIPS['Precision']}
                  </div>
                </th>
                <th className="px-6 py-4 font-semibold group cursor-help relative">
                  <div className="flex items-center gap-1.5">
                    Hallucination
                    <HelpCircle className="w-3 h-3 opacity-50" />
                  </div>
                  <div className="absolute hidden group-hover:block top-full mt-2 bg-slate-800 text-slate-300 text-xs p-2 rounded border border-white/10 w-48 normal-case font-normal shadow-xl">
                    {SCORE_TOOLTIPS['Hallucination']}
                  </div>
                </th>
                <th className="px-6 py-4 font-semibold group cursor-help relative">
                  <div className="flex items-center gap-1.5">
                    Overall
                    <HelpCircle className="w-3 h-3 opacity-50" />
                  </div>
                  <div className="absolute hidden group-hover:block top-full mt-2 right-0 bg-slate-800 text-slate-300 text-xs p-2 rounded border border-white/10 w-48 normal-case font-normal shadow-xl">
                    {SCORE_TOOLTIPS['Overall']}
                  </div>
                </th>
              </tr>
            </thead>
            <motion.tbody 
              variants={container} 
              initial="hidden" 
              animate="show"
              className="divide-y divide-white/5"
            >
              {evals.length === 0 && !loading ? (
                <tr>
                  <td colSpan="7" className="px-6 py-16 text-center">
                    <div className="flex flex-col items-center justify-center text-slate-500">
                      <Activity className="w-12 h-12 mb-4 opacity-50" />
                      <p className="font-medium">No evaluations recorded yet.</p>
                      <p className="text-xs mt-1">Run a deep research query first to generate quality scores.</p>
                    </div>
                  </td>
                </tr>
              ) : (
                evals.map((ev, i) => (
                  <motion.tr variants={item} key={i} className="hover:bg-white/5 transition-colors duration-150">
                    <td className="px-6 py-4 whitespace-nowrap text-slate-400 font-mono text-xs">
                      {new Date(ev.created_at).toLocaleDateString()} <span className="opacity-50">{new Date(ev.created_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                    </td>
                    <td className="px-6 py-4 max-w-[200px] truncate text-slate-200 font-medium" title={ev.question}>
                      {ev.question}
                    </td>
                    <td className={`px-6 py-4 font-mono font-medium ${getColor(ev.faithfulness)}`}>{formatScore(ev.faithfulness)}</td>
                    <td className={`px-6 py-4 font-mono font-medium ${getColor(ev.answer_relevance)}`}>{formatScore(ev.answer_relevance)}</td>
                    <td className={`px-6 py-4 font-mono font-medium ${getColor(ev.context_precision)}`}>{formatScore(ev.context_precision)}</td>
                    <td className={`px-6 py-4 font-mono font-medium ${getColor(ev.hallucination_risk, true)}`}>{formatScore(ev.hallucination_risk)}</td>
                    <td className={`px-6 py-4 font-mono font-bold text-base ${getColor(ev.overall_score)}`}>{formatScore(ev.overall_score)}</td>
                  </motion.tr>
                ))
              )}
            </motion.tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
