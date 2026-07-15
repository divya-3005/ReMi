import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import { UploadCloud, FileText, Trash2, FileX, Loader2, Calendar, Layers, HardDrive, AlertTriangle } from 'lucide-react'
import { useToast } from '../contexts/ToastContext'

const MAX_FILE_SIZE = 50 * 1024 * 1024 // 50MB

function DeleteConfirmDialog({ docName, onConfirm, onCancel }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onCancel}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 10 }}
        className="glass-card p-8 max-w-md w-full mx-4 shadow-2xl border border-white/10"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-red-500/10 flex items-center justify-center flex-shrink-0">
            <AlertTriangle className="w-5 h-5 text-red-400" />
          </div>
          <h3 className="text-white font-semibold text-lg">Delete Document?</h3>
        </div>
        <p className="text-slate-400 text-sm mb-6 ml-[52px]">
          This will permanently remove <span className="text-white font-medium">"{docName}"</span> and all its indexed data. This action cannot be undone.
        </p>
        <div className="flex justify-end gap-3">
          <button
            onClick={onCancel}
            className="px-5 py-2.5 rounded-xl text-slate-300 hover:bg-white/5 transition-colors font-medium text-sm"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="px-5 py-2.5 rounded-xl bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30 transition-colors font-medium text-sm"
          >
            Delete
          </button>
        </div>
      </motion.div>
    </motion.div>
  )
}

export default function DocumentPanel() {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState(null)
  const fileInputRef = useRef(null)
  const toast = useToast()

  const fetchDocs = async () => {
    setLoading(true)
    try {
      const res = await axios.get('/api/documents')
      setDocuments(res.data)
    } catch (err) {
      toast.error('Failed to load documents. Is the backend running?')
    }
    setLoading(false)
  }

  useEffect(() => {
    fetchDocs()
  }, [])

  const validateAndUpload = async (file) => {
    if (!file) return

    // Validate file type
    const validTypes = ['.pdf', '.txt']
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!validTypes.includes(ext)) {
      toast.error(`Unsupported file type "${ext}". Please upload a PDF or TXT file.`)
      return
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
      toast.error(`File too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum size is 50MB.`)
      return
    }

    const formData = new FormData()
    formData.append('file', file)

    setUploading(true)
    try {
      const res = await axios.post('/api/documents/ingest', formData)
      toast.success(`"${file.name}" uploaded successfully! (${res.data.chunk_count} searchable sections created)`)
      await fetchDocs()
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || 'Unknown error'
      toast.error(`Upload failed: ${detail}`)
    }
    setUploading(false)
  }

  const handleFileUpload = async (e) => {
    await validateAndUpload(e.target.files[0])
    e.target.value = null
  }

  const handleDrop = async (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    await validateAndUpload(file)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setDragOver(false)
  }

  const handleDelete = async (doc) => {
    try {
      await axios.delete(`/api/documents/${doc.id}`)
      toast.success(`"${doc.filename}" deleted successfully.`)
      setDeleteTarget(null)
      await fetchDocs()
    } catch (err) {
      toast.error('Failed to delete document.')
      setDeleteTarget(null)
    }
  }

  const formatDate = (dateStr) => {
    try {
      const d = new Date(dateStr)
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    } catch {
      return 'Unknown'
    }
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
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-white">My Documents</h2>
        <p className="text-sm text-slate-500 mt-1">Upload PDF or TXT files to start researching them with AI.</p>
      </div>

      {/* Upload Zone with real drag-and-drop */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`glass-card border-dashed border-2 p-8 text-center transition-all duration-200 ease-out ${
          dragOver
            ? 'border-emerald-500/60 bg-emerald-500/5 scale-[1.01]'
            : 'border-slate-700 hover:bg-white/5 hover:border-slate-600'
        }`}
      >
        {uploading ? (
          <div className="flex flex-col items-center justify-center space-y-3 text-emerald-400">
            <Loader2 className="w-8 h-8 animate-spin" />
            <span className="font-medium text-sm">Processing document...</span>
            <span className="text-xs text-slate-500">Extracting text, creating searchable sections, building index...</span>
          </div>
        ) : (
          <label className="cursor-pointer flex flex-col items-center justify-center space-y-3">
            <div className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${
              dragOver ? 'bg-emerald-500/20 text-emerald-400' : 'bg-emerald-500/10 text-emerald-500'
            }`}>
              <UploadCloud className="w-6 h-6" />
            </div>
            <div className="text-slate-400 text-sm">
              {dragOver ? (
                <span className="text-emerald-400 font-medium">Drop your file here</span>
              ) : (
                'Drag & drop a file here, or click to browse'
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".pdf,.txt"
              onChange={handleFileUpload}
            />
            {!dragOver && (
              <div className="mt-4 px-5 py-2.5 bg-[#1e2535] hover:bg-slate-800 border border-white/5 text-slate-300 rounded-xl font-medium transition-colors duration-150 ease-out text-sm">
                Select File
              </div>
            )}
            <div className="text-xs text-slate-600 mt-2">Supports PDF and TXT files up to 50MB</div>
          </label>
        )}
      </div>

      {/* Document List */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs uppercase tracking-widest text-slate-400">Your Documents ({documents.length})</h3>
        </div>

        {loading ? (
          <div className="grid gap-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="glass-card p-5 flex justify-between items-center animate-pulse">
                <div className="space-y-3 flex-1">
                  <div className="h-4 bg-slate-800 rounded w-1/3"></div>
                  <div className="h-3 bg-slate-800 rounded w-1/4"></div>
                </div>
                <div className="h-8 w-16 bg-slate-800 rounded"></div>
              </div>
            ))}
          </div>
        ) : documents.length === 0 ? (
          <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
            <FileX className="w-12 h-12 text-slate-600 mb-4" />
            <h3 className="text-slate-300 font-medium mb-1">No documents yet</h3>
            <p className="text-slate-500 text-sm">Upload a PDF or TXT file above to get started.</p>
          </div>
        ) : (
          <motion.div variants={container} initial="hidden" animate="show" className="grid gap-4">
            {documents.map((doc) => (
              <motion.div variants={item} key={doc.id} className="glass-card p-5 flex justify-between items-center group hover:border-emerald-500/30 transition-colors duration-150">
                <div className="flex items-start gap-4 flex-1 min-w-0">
                  <div className="p-2 bg-slate-800/50 rounded-lg text-slate-400 flex-shrink-0">
                    <FileText className="w-5 h-5" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="font-medium text-slate-200 truncate">{doc.filename}</div>
                    <div className="flex items-center flex-wrap gap-x-4 gap-y-1 mt-1.5 text-xs text-slate-500">
                      <span className="flex items-center gap-1">
                        <Layers className="w-3 h-3" />
                        {doc.chunk_count} sections
                      </span>
                      <span className="flex items-center gap-1">
                        <HardDrive className="w-3 h-3" />
                        {(doc.file_size_bytes / 1024).toFixed(1)} KB
                      </span>
                      {doc.page_count && (
                        <span>{doc.page_count} pages</span>
                      )}
                      {doc.ingested_at && (
                        <span className="flex items-center gap-1">
                          <Calendar className="w-3 h-3" />
                          {formatDate(doc.ingested_at)}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => setDeleteTarget(doc)}
                  className="p-2 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors duration-150 flex-shrink-0 ml-3"
                  title="Delete Document"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </motion.div>
            ))}
          </motion.div>
        )}
      </div>

      {/* Delete Confirmation Dialog */}
      <AnimatePresence>
        {deleteTarget && (
          <DeleteConfirmDialog
            docName={deleteTarget.filename}
            onConfirm={() => handleDelete(deleteTarget)}
            onCancel={() => setDeleteTarget(null)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}
