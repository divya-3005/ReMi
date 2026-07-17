<div align="center">
  <h1>ReMi — Research Mind</h1>
  <p><strong>Agentic Research Assistant with Zero-Hallucination Traceability</strong></p>
</div>

ReMi is a compound AI system designed to solve the hardest problem in enterprise Retrieval-Augmented Generation (RAG): **hallucinations and untraceable claims**. 

Unlike standard RAG pipelines that simply "retrieve and synthesize," ReMi utilizes a **multi-agent feedback loop** and a deterministic **difflib grounding engine** to guarantee that every single factual claim in its output is strictly mapped to an exact character span in the source documents. If the system fails to ground its claims, it detects the failure and autonomously reformulates its search strategy.

## 🌟 Key Features & Engineering Highlights

### 1. Multi-Agent Feedback Loop
ReMi orchestrates a pipeline of specialized agents. If the final report falls below configured quality thresholds (e.g., low citation coverage), the **Evaluator** triggers a retry, passing the exact failure metrics back to the **Planner** to autonomously reformulate the search strategy.
* **Planner**: Decomposes complex queries into focused sub-questions and generates HyDE variants.
* **Researcher**: Executes concurrent retrievals against the vector store.
* **Analyzer**: A high-throughput classification agent that filters out irrelevant chunks.
* **Synthesizer**: Compiles findings into a continuous, cohesive report.
* **Grounder**: A deterministic algorithm mapping generated claims to absolute source text.
* **Evaluator**: Scores the final output on coverage, utilization, relevance, and hallucination risk.

### 2. Deterministic Traceability
LLMs are excellent at synthesizing, but terrible at verifying their own truthfulness (LLM-as-a-judge often fails). ReMi sidesteps this by using a deterministic `difflib` sequence matching algorithm in its Grounder agent. Every `[^1]` footnote in the UI is a physical link to a verified contiguous character span in the underlying document.

### 3. Thread-Safe Hybrid Retrieval
The `HybridStore` implements **Reciprocal Rank Fusion (RRF)** over FAISS (dense) and BM25 (sparse) embeddings. It is engineered for concurrent access, utilizing re-entrant locks (`RLock`) to guarantee thread safety while ensuring high-latency network embedding calls occur *outside* the lock to prevent bottlenecking.

### 4. Dual-Model Architecture for Cost/Latency
ReMi abstracts model interactions, routing tasks based on their requirements:
* **Groq (Llama 3)**: Used by the Analyzer for sub-second, high-throughput classification and chunk filtering.
* **Google Gemini**: Used by the Planner and Synthesizer for high-reasoning, heavy context-window tasks.

### 5. Dependency-Free, Premium Frontend
The frontend abandons heavy reactive frameworks in favor of a highly optimized, vanilla HTML/CSS/JS architecture. It features a strict typographic system (utilizing optical sizing via `Fraunces`), sub-pixel CSS physics for tactile feedback, and CSS-only skeleton loaders—achieving an editorial, FAANG-caliber aesthetic without the bloat.

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
git clone https://github.com/divya-3005/ReMi.git
cd ReMi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory and add your API keys:
```bash
cp .env.example .env
```
Ensure you provide:
- `GEMINI_API_KEY`
- `GROQ_API_KEY`

### 3. Start the Backend API
Start the FastAPI server (runs on `localhost:8000`):
```bash
uvicorn src.api.main:app --reload
```

### 4. Start the Frontend
In a new terminal window, serve the frontend on `localhost:8080`:
```bash
python3 -m http.server 8080 --directory frontend/
```
Navigate to `http://localhost:8080` in your browser.

---

## 💻 CLI Usage
For debugging and terminal usage, ReMi includes a full Typer CLI:

```bash
# Upload a document to the local vector store
python cli.py upload /path/to/document.pdf

# Run a research query
python cli.py ask "What were the main causes of the 2008 financial crisis?"

# Get raw JSON output containing full trace evaluation
python cli.py ask "What is quantitative easing?" --json
```

## 🏗️ Project Structure
* `src/agent/` - LLM agents and workflow orchestration.
* `src/api/` - FastAPI REST endpoints.
* `src/evaluation/` - Quality scoring and LLM evaluation metrics.
* `src/genai/` - Abstracted LLM client wrappers (Groq/Gemini).
* `src/grounding/` - Deterministic difflib citation matching.
* `src/ingestion/` - PDF/TXT parsing and semantic chunking.
* `src/vectorstore/` - Hybrid FAISS/BM25 retrieval engine.
* `frontend/` - Premium vanilla JS/CSS UI.
* `tests/` - Comprehensive Pytest suite (150+ tests).
