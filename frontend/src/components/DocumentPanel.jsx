import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { motion } from 'framer-motion'
import { UploadCloud, FileText, Trash2, FileX, Loader2 } from 'lucide-react'

export default function DocumentPanel() {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)

  const fetchDocs = async () => {
    setLoading(true)
    try {
      const res = await axios.get('/api/documents')
      setDocuments(res.data)
    } catch (err) {
      console.error(err)
    }
    setLoading(false)
  }

  useEffect(() => {
    fetchDocs()
  }, [])

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    setUploading(true)
    try {
      await axios.post('/api/documents/ingest', formData)
      await fetchDocs()
    } catch (err) {
      console.error(err)
      alert("Upload failed")
    }
    setUploading(false)
    e.target.value = null // reset input
  }

  const handleDelete = async (docId) => {
    try {
      await axios.delete(`/api/documents/${docId}`)
      await fetchDocs()
    } catch (err) {
      console.error(err)
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
      <h2 className="text-xl font-semibold text-white">Documents</h2>

      {/* Upload Zone */}
      <div className="glass-card border-dashed border-2 border-slate-700 p-8 text-center hover:bg-white/5 transition-colors duration-150 ease-out">
        {uploading ? (
          <div className="flex flex-col items-center justify-center space-y-3 text-emerald-400">
            <Loader2 className="w-8 h-8 animate-spin" />
            <span className="font-medium text-sm">Ingesting document...</span>
          </div>
        ) : (
          <label className="cursor-pointer flex flex-col items-center justify-center space-y-3">
            <div className="w-12 h-12 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500">
              <UploadCloud className="w-6 h-6" />
            </div>
            <div className="text-slate-400 text-sm">Drag & drop or click to upload PDF/TXT</div>
            <input type="file" className="hidden" accept=".pdf,.txt" onChange={handleFileUpload} />
            <div className="mt-4 px-5 py-2.5 bg-[#1e2535] hover:bg-slate-800 border border-white/5 text-slate-300 rounded-xl font-medium transition-colors duration-150 ease-out">
              Select File
            </div>
          </label>
        )}
      </div>

      {/* Document List */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs uppercase tracking-widest text-slate-400">Ingested Documents ({documents.length})</h3>
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
            <p className="text-slate-500 text-sm">Upload a PDF or TXT file to get started.</p>
          </div>
        ) : (
          <motion.div variants={container} initial="hidden" animate="show" className="grid gap-4">
            {documents.map((doc) => (
              <motion.div variants={item} key={doc.id} className="glass-card p-5 flex justify-between items-center group hover:border-emerald-500/30 transition-colors duration-150">
                <div className="flex items-start gap-4">
                  <div className="p-2 bg-slate-800/50 rounded-lg text-slate-400">
                    <FileText className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="font-medium text-slate-200">{doc.filename}</div>
                    <div className="text-sm text-slate-500 mt-1">
                      {doc.chunk_count} chunks • {(doc.file_size_bytes / 1024).toFixed(1)} KB
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => handleDelete(doc.id)}
                  className="p-2 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors duration-150"
                  title="Delete Document"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </motion.div>
            ))}
          </motion.div>
        )}
      </div>
    </div>
  )
}
