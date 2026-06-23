import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FileText, Search, MessageSquare, Microscope, Activity } from 'lucide-react'
import DocumentPanel from './components/DocumentPanel'
import SearchPanel from './components/SearchPanel'
import AskPanel from './components/AskPanel'
import ResearchPanel from './components/ResearchPanel'
import EvaluationsPanel from './components/EvaluationsPanel'

const TABS = [
  { id: 'documents', label: 'Documents', icon: FileText },
  { id: 'search', label: 'Search', icon: Search },
  { id: 'ask', label: 'Ask', icon: MessageSquare },
  { id: 'research', label: 'Research', icon: Microscope },
  { id: 'evaluations', label: 'Evaluations', icon: Activity }
];

export default function App() {
  const [activeTab, setActiveTab] = useState('documents')

  return (
    <div className="flex h-screen bg-transparent text-slate-200 font-sans">
      {/* Sidebar */}
      <div className="w-64 border-r border-white/5 flex flex-col bg-slate-950/20 backdrop-blur-md">
        <div className="p-6 border-b border-white/5">
          <h1 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <Microscope className="w-6 h-6 text-emerald-500" />
            Research<span className="text-emerald-500">Mind</span>
          </h1>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-150 font-medium ${
                  isActive
                    ? 'bg-gradient-to-r from-emerald-500/20 to-transparent border-l-2 border-emerald-500 text-emerald-400'
                    : 'text-slate-400 hover:bg-white/5 hover:text-slate-200 border-l-2 border-transparent'
                }`}
              >
                <Icon className="w-5 h-5" />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto relative">
        <div className="max-w-5xl mx-auto p-8 h-full">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
              className="h-full"
            >
              {activeTab === 'documents' && <DocumentPanel />}
              {activeTab === 'search' && <SearchPanel />}
              {activeTab === 'ask' && <AskPanel />}
              {activeTab === 'research' && <ResearchPanel />}
              {activeTab === 'evaluations' && <EvaluationsPanel />}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  )
}
