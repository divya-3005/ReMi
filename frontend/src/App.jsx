import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import { FileText, Search, MessageSquare, Microscope, Activity, Wifi, WifiOff, Upload, HelpCircle, Zap } from 'lucide-react'
import { ToastProvider } from './contexts/ToastContext'
import DocumentPanel from './components/DocumentPanel'
import SearchPanel from './components/SearchPanel'
import AskPanel from './components/AskPanel'
import ResearchPanel from './components/ResearchPanel'
import EvaluationsPanel from './components/EvaluationsPanel'

const TABS = [
  { id: 'documents', label: 'My Documents', desc: 'Upload & manage files', icon: FileText },
  { id: 'search', label: 'Search', desc: 'Find info in your docs', icon: Search },
  { id: 'ask', label: 'Ask AI', desc: 'Get instant answers', icon: MessageSquare },
  { id: 'research', label: 'Deep Research', desc: 'Comprehensive reports', icon: Microscope },
  { id: 'evaluations', label: 'Quality Scores', desc: 'Track answer quality', icon: Activity }
];

function WelcomeScreen({ onGetStarted }) {
  const steps = [
    { icon: Upload, title: 'Upload a Document', desc: 'Drop in a PDF or TXT file. We\'ll break it into searchable pieces automatically.', color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
    { icon: MessageSquare, title: 'Ask Questions', desc: 'Ask anything in plain English. The AI finds relevant passages and generates a cited answer.', color: 'text-blue-400', bg: 'bg-blue-500/10' },
    { icon: Microscope, title: 'Run Deep Research', desc: 'For complex questions, our AI breaks them down, researches each part, and writes a full report.', color: 'text-violet-400', bg: 'bg-violet-500/10' },
  ]

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8 max-w-2xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-12"
      >
        <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-violet-500/20 border border-white/10 flex items-center justify-center mx-auto mb-6">
          <Zap className="w-10 h-10 text-emerald-400" />
        </div>
        <h1 className="text-3xl font-bold text-white mb-3">Welcome to Research<span className="text-emerald-400">Mind</span></h1>
        <p className="text-slate-400 text-lg max-w-md mx-auto">
          Upload your documents and let AI help you find answers, run research, and generate cited reports.
        </p>
      </motion.div>

      <div className="grid gap-6 w-full mb-10">
        {steps.map((step, i) => {
          const Icon = step.icon
          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: 0.15 * (i + 1) }}
              className="glass-card p-6 flex items-start gap-5 group hover:border-white/10 transition-colors"
            >
              <div className={`w-12 h-12 rounded-xl ${step.bg} flex items-center justify-center flex-shrink-0`}>
                <Icon className={`w-6 h-6 ${step.color}`} />
              </div>
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Step {i + 1}</span>
                </div>
                <h3 className="text-white font-semibold mb-1">{step.title}</h3>
                <p className="text-slate-400 text-sm leading-relaxed">{step.desc}</p>
              </div>
            </motion.div>
          )
        })}
      </div>

      <motion.button
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.7 }}
        onClick={onGetStarted}
        className="bg-emerald-500 hover:bg-emerald-400 text-black px-8 py-3.5 rounded-xl font-semibold transition-colors duration-150 shadow-lg shadow-emerald-500/20 flex items-center gap-2"
      >
        <Upload className="w-5 h-5" />
        Upload Your First Document
      </motion.button>
    </div>
  )
}

function HealthIndicator({ isHealthy }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      {isHealthy ? (
        <>
          <div className="relative">
            <span className="w-2 h-2 bg-emerald-500 rounded-full block"></span>
            <span className="w-2 h-2 bg-emerald-500 rounded-full block absolute inset-0 animate-ping opacity-50"></span>
          </div>
          <span className="text-slate-500">Connected</span>
        </>
      ) : (
        <>
          <span className="w-2 h-2 bg-red-500 rounded-full block"></span>
          <span className="text-red-400/70">Offline</span>
        </>
      )}
    </div>
  )
}

function AppContent() {
  const [activeTab, setActiveTab] = useState('documents')
  const [backendHealthy, setBackendHealthy] = useState(true)
  const [hasDocuments, setHasDocuments] = useState(null) // null = loading

  // Health check polling
  useEffect(() => {
    const check = async () => {
      try {
        const res = await axios.get('/api/health')
        setBackendHealthy(res.status === 200)
      } catch {
        setBackendHealthy(false)
      }
    }
    check()
    const interval = setInterval(check, 15000)
    return () => clearInterval(interval)
  }, [])

  // Check if documents exist (for welcome screen)
  useEffect(() => {
    const checkDocs = async () => {
      try {
        const res = await axios.get('/api/documents')
        setHasDocuments(res.data.length > 0)
      } catch {
        setHasDocuments(false)
      }
    }
    checkDocs()
  }, [activeTab]) // re-check when switching tabs

  const showWelcome = hasDocuments === false && activeTab === 'documents'

  return (
    <div className="flex h-screen bg-transparent text-slate-200 font-sans">
      {/* Sidebar */}
      <div className="w-72 border-r border-white/5 flex flex-col bg-slate-950/20 backdrop-blur-md">
        <div className="p-6 border-b border-white/5">
          <h1 className="text-xl font-bold text-white tracking-tight flex items-center gap-3">
            <img src="/logo.png" alt="ResearchMind Logo" className="w-8 h-8 drop-shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
            Research<span className="text-emerald-500">Mind</span>
          </h1>
          <p className="text-[11px] text-slate-500 mt-1.5">AI-Powered Document Research</p>
        </div>
        <nav className="flex-1 p-4 space-y-1.5 overflow-y-auto">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-150 text-left ${
                  isActive
                    ? 'bg-gradient-to-r from-emerald-500/20 to-transparent border-l-2 border-emerald-500 text-emerald-400'
                    : 'text-slate-400 hover:bg-white/5 hover:text-slate-200 border-l-2 border-transparent'
                }`}
              >
                <Icon className="w-5 h-5 flex-shrink-0" />
                <div>
                  <div className="font-medium text-sm">{tab.label}</div>
                  <div className={`text-[11px] ${isActive ? 'text-emerald-500/60' : 'text-slate-600'}`}>{tab.desc}</div>
                </div>
              </button>
            );
          })}
        </nav>
        <div className="p-4 border-t border-white/5">
          <HealthIndicator isHealthy={backendHealthy} />
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto relative">
        <div className="max-w-5xl mx-auto p-8 h-full">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab + (showWelcome ? '-welcome' : '')}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
              className="h-full flex flex-col"
            >
              {showWelcome ? (
                <WelcomeScreen onGetStarted={() => setHasDocuments(null)} />
              ) : (
                <>
                  {activeTab === 'documents' && <DocumentPanel />}
                  {activeTab === 'search' && <SearchPanel />}
                  {activeTab === 'ask' && <AskPanel />}
                  {activeTab === 'research' && <ResearchPanel />}
                  {activeTab === 'evaluations' && <EvaluationsPanel />}
                </>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <ToastProvider>
      <AppContent />
    </ToastProvider>
  )
}
