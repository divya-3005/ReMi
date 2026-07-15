import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import axios from 'axios'
import './index.css'

// If VITE_API_URL is set (e.g. in Vercel), axios will route all requests there.
// If it's undefined (e.g. local dev), axios will use relative paths (which Vite proxies).
axios.defaults.baseURL = import.meta.env.VITE_API_URL || '';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
