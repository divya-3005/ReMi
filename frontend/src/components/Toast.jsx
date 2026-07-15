import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle2, XCircle, Info, X } from 'lucide-react'

const iconMap = {
  success: CheckCircle2,
  error: XCircle,
  info: Info,
}

const colorMap = {
  success: {
    bg: 'bg-emerald-500/10 border-emerald-500/30',
    icon: 'text-emerald-400',
    text: 'text-emerald-200',
  },
  error: {
    bg: 'bg-red-500/10 border-red-500/30',
    icon: 'text-red-400',
    text: 'text-red-200',
  },
  info: {
    bg: 'bg-blue-500/10 border-blue-500/30',
    icon: 'text-blue-400',
    text: 'text-blue-200',
  },
}

export default function Toast({ message, type = 'info', onClose }) {
  const [visible, setVisible] = useState(true)
  const Icon = iconMap[type] || Info
  const colors = colorMap[type] || colorMap.info

  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(false)
      setTimeout(onClose, 300)
    }, 3700)
    return () => clearTimeout(timer)
  }, [onClose])

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 10, scale: 0.95 }}
          transition={{ duration: 0.25, ease: 'easeOut' }}
          className={`pointer-events-auto flex items-center gap-3 px-5 py-3.5 rounded-xl border backdrop-blur-xl shadow-2xl min-w-[300px] max-w-[450px] ${colors.bg}`}
        >
          <Icon className={`w-5 h-5 flex-shrink-0 ${colors.icon}`} />
          <span className={`text-sm font-medium flex-1 ${colors.text}`}>{message}</span>
          <button
            onClick={() => { setVisible(false); setTimeout(onClose, 300); }}
            className="text-slate-500 hover:text-slate-300 transition-colors flex-shrink-0"
          >
            <X className="w-4 h-4" />
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
