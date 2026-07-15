<div align="center">
  <img src="frontend/public/logo.png" alt="ResearchMind Logo" width="120"/>
  <h1>ResearchMind 🧠</h1>

  <p><strong>An autonomous, multi-agent RAG platform that reads your documents, runs deep research workflows, and generates rigorously cited reports.</strong></p>

  <p>
    <a href="#-features">Features</a> •
    <a href="#-quick-start-5-minutes">Quick Start</a> •
    <a href="#%EF%B8%8F-architecture">Architecture</a> •
    <a href="#-production-deployment">Deployment</a>
  </p>

  <p>
    <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python Version" />
    <img src="https://img.shields.io/badge/React-19-61dafb?logo=react&logoColor=black" alt="React" />
    <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
  </p>
</div>

---

Unlike simple chat wrappers that hallucinate answers, ResearchMind explicitly grounds every claim it makes to exact character spans in your source text. It evaluates its own performance quantitatively and runs inside a premium, glass-morphism web interface.

## ✨ Features

- **Multi-Step Agentic Workflow**: Complex questions are broken down by a *Planner Agent*, investigated by specialized *Researcher Agents*, and filtered by an *Analyzer* before final synthesis.
- **Advanced RAG Capabilities**: Features *Semantic Chunking* (recursive text splitting) to preserve paragraph context, *Parallel Execution* for massive speedups, and *Query Expansion (HyDE)* to maximize retrieval accuracy.
- **Strict Evidence Grounding**: A custom NLP layer enforces that every sentence in the final report has a verifiable citation back to your uploaded PDF/TXT files.
- **Automated RAG Evaluation**: Every research run generates quality scores for **Faithfulness**, **Answer Relevance**, **Context Precision**, and **Hallucination Risk**.
- **Lightning Fast & Memory Efficient**: By offloading embeddings to the external **Google Gemini API**, ReMi runs complex semantic vector searches locally via FAISS entirely within 512MB RAM constraints, guaranteeing 100% uptime on free tiers.
- **Beautiful UI**: A responsive React/Vite frontend with dynamic toast notifications, real-time polling, and detailed metric dashboards.

---

## 🚀 Quick Start (5 Minutes)

Assume a fresh machine (MacOS/Linux) with Python 3.11+ and Node.js installed.

### 1. Setup Backend
```bash
git clone https://github.com/divyasingh1/ReMi.git
cd ReMi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Start the Platform
```bash
# Start FastAPI backend (Port 8000)
python cli.py serve &

# Start React frontend (Port 5173)
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:5173` to view the application!

---

## 🏗️ Architecture

```mermaid
flowchart TD
    UI[React Web App] -->|User Query| API[FastAPI Backend]
    API --> Planner[Planner Agent]
    
    subgraph Multi-Agent Research Workflow
        Planner -->|Decomposes Query| SubQs[Sub-Questions]
        SubQs --> Researcher[Researcher Agents]
        Researcher -->|Search FAISS Index| VectorStore[(FAISS + Gemini API)]
        VectorStore -->|Return Chunks| Analyzer[Analyzer Agent]
        Analyzer -->|Filter & Rank| Synthesizer[Synthesizer Agent]
    end
    
    Synthesizer -->|Draft Report| Grounding[Grounding Layer]
    Grounding -->|Link Claims to Spans| Evaluator[Evaluation Layer]
    Evaluator -->|Score Quality| FinalReport[Final Markdown Report]
    FinalReport --> API
    API --> UI
```

---

## ☁️ Production Deployment

ResearchMind is designed to be easily deployed for free to showcase in portfolios.

### 1. Backend (Render.com - Free Tier)
Deploy the root folder as a Docker Web Service on Render. Because we offloaded heavy Machine Learning embeddings to the external Google Gemini API, it fits comfortably within Render's 512MB free RAM limit and never crashes.
*Note: Render's free tier is ephemeral, meaning uploaded documents reset when the server sleeps. This creates a perfect, clean-slate demo environment for recruiters!*

### 2. Frontend (Vercel - Free Tier)
Deploy the `frontend/` folder to Vercel as a Vite project. 
To securely link it to your backend without exposing the URL in your code:
- Go to Vercel Project Settings → Environment Variables
- Add `VITE_API_URL` and set it to your Render URL (e.g., `https://remi-backend-xyz.onrender.com`).

---

## 💻 CLI Usage

You can orchestrate the entire pipeline directly from the terminal without the web UI:

```bash
# Ingest a document into the FAISS vector store
python cli.py ingest /path/to/document.pdf

# Run a full agentic research query
python cli.py research "What are the core capabilities of the ingested document?"

# Check document store stats
python cli.py stats
```

---

## 📊 Evaluation Metrics

ResearchMind doesn't just guess quality; it calculates it:
- **Faithfulness:** Ratio of generated sentences that map back to a source chunk.
- **Answer Relevance:** Cosine similarity between query and final report.
- **Context Precision:** Percentage of retrieved chunks that were actually useful.
- **Hallucination Risk:** Inverse of faithfulness (1.0 - Faithfulness, lower is better).
