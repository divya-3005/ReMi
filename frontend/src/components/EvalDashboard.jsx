import React from 'react';
import { motion } from 'framer-motion';
import { HelpCircle } from 'lucide-react';

const SCORE_TOOLTIPS = {
  'Faithfulness': 'Did the AI stick to facts from your documents?',
  'Answer Relevance': 'Did the answer actually address your question?',
  'Context Precision': 'Were the right parts of documents found?',
  'Hallucination Risk': 'Did the AI make anything up? (Lower is better)'
}

const MetricBar = ({ label, value, delay }) => {
  const percentage = Math.round(value * 100);
  
  let colorClass = 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]';
  let trackClass = 'bg-red-500/10';
  
  if (value >= 0.7) {
    colorClass = 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]';
    trackClass = 'bg-emerald-500/10';
  } else if (value >= 0.4) {
    colorClass = 'bg-yellow-500 shadow-[0_0_10px_rgba(234,179,8,0.5)]';
    trackClass = 'bg-yellow-500/10';
  }

  // For hallucination risk, lower is better. Reverse colors.
  if (label.includes('Risk')) {
    if (value <= 0.3) {
      colorClass = 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]';
      trackClass = 'bg-emerald-500/10';
    } else if (value <= 0.6) {
      colorClass = 'bg-yellow-500 shadow-[0_0_10px_rgba(234,179,8,0.5)]';
      trackClass = 'bg-yellow-500/10';
    } else {
      colorClass = 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]';
      trackClass = 'bg-red-500/10';
    }
  }

  const tooltip = SCORE_TOOLTIPS[label];

  return (
    <div className="mb-5 relative group">
      <div className="flex justify-between mb-2">
        <span className="text-sm font-medium text-slate-300 flex items-center gap-1.5 cursor-help">
          {label}
          {tooltip && <HelpCircle className="w-3 h-3 text-slate-500" />}
        </span>
        <span className="text-sm font-mono font-medium text-slate-300">{percentage}%</span>
      </div>
      
      {/* Tooltip */}
      {tooltip && (
        <div className="absolute bottom-full left-0 mb-1 bg-slate-800 border border-white/10 rounded-lg p-2 text-xs text-slate-300 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 w-64 shadow-xl">
          {tooltip}
        </div>
      )}

      <div className={`w-full rounded-full h-2 ${trackClass} overflow-hidden`}>
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 1, delay, ease: "easeOut" }}
          className={`h-2 rounded-full ${colorClass}`} 
        ></motion.div>
      </div>
    </div>
  );
};

export default function EvalDashboard({ evalResult }) {
  if (!evalResult) return null;

  return (
    <div className="mt-12 pt-8 border-t border-white/5 relative">
      <h3 className="text-xs uppercase tracking-widest text-slate-400 font-semibold mb-6">Evaluation Metrics</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="col-span-1 md:col-span-2 glass-card p-6 shadow-lg">
          <MetricBar label="Faithfulness" value={evalResult.faithfulness} delay={0.1} />
          <MetricBar label="Answer Relevance" value={evalResult.answer_relevance} delay={0.2} />
          <MetricBar label="Context Precision" value={evalResult.context_precision} delay={0.3} />
          <MetricBar label="Hallucination Risk" value={evalResult.hallucination_risk} delay={0.4} />
        </div>
        
        <div className="col-span-1 glass-card p-6 flex flex-col items-center justify-center relative overflow-hidden shadow-lg group">
          <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
          <h3 className="text-xs uppercase tracking-widest text-slate-400 font-semibold mb-4 z-10 flex items-center gap-1.5 cursor-help">
            Overall Quality
            <HelpCircle className="w-3 h-3" />
          </h3>
          
          <div className="absolute top-12 left-1/2 -translate-x-1/2 w-48 bg-slate-800 border border-white/10 rounded-lg p-2 text-xs text-slate-300 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-20 shadow-xl text-center">
            Combined quality score across all metrics.
          </div>

          <div className="relative w-36 h-36 flex items-center justify-center rounded-full bg-[#080c14] border border-white/5 shadow-[inset_0_0_20px_rgba(0,0,0,0.5)] z-10">
            <svg className="absolute inset-0 w-full h-full transform -rotate-90">
              <circle cx="72" cy="72" r="68" stroke="currentColor" strokeWidth="2" fill="transparent" className="text-slate-800" />
              <motion.circle 
                initial={{ strokeDashoffset: 427 }}
                animate={{ strokeDashoffset: 427 - (427 * evalResult.overall_score) }}
                transition={{ duration: 1.5, ease: "easeOut" }}
                cx="72" cy="72" r="68" 
                stroke="currentColor" strokeWidth="4" 
                fill="transparent" 
                strokeDasharray="427"
                strokeLinecap="round"
                className="text-emerald-500 drop-shadow-[0_0_8px_rgba(16,185,129,0.8)]" 
              />
            </svg>
            <span className="text-5xl font-bold text-white tracking-tighter">
              {Math.round(evalResult.overall_score * 100)}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-6 text-center z-10">
            Weighted composite of all quality metrics
          </p>
        </div>
      </div>
    </div>
  );
}
