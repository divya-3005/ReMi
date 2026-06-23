import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { Activity, RefreshCw } from 'lucide-react';

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
        <h2 className="text-xl font-semibold text-white">Evaluation History</h2>
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
            { label: 'Avg Faithfulness', val: summary.averages.faithfulness },
            { label: 'Avg Relevance', val: summary.averages.answer_relevance },
            { label: 'Avg Precision', val: summary.averages.context_precision },
            { label: 'Avg Hallucination', val: summary.averages.hallucination_risk, reverse: true },
            { label: 'Overall Quality', val: summary.averages.overall_score }
          ].map((s, i) => (
            <div key={i} className="glass-card p-5 text-center shadow-lg">
              <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-2">{s.label}</div>
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
                <th className="px-6 py-4 font-semibold">Faithfulness</th>
                <th className="px-6 py-4 font-semibold">Relevance</th>
                <th className="px-6 py-4 font-semibold">Precision</th>
                <th className="px-6 py-4 font-semibold">Hallucination</th>
                <th className="px-6 py-4 font-semibold">Overall</th>
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
                      <p className="text-xs mt-1">Run a research query first to generate metrics.</p>
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
