# ReMi — Research Mind

ReMi is an agentic research assistant that answers questions from your uploaded documents. 
Its core design constraint is absolute traceability: every factual claim in the generated report is linked to an exact character span in the source documents.

## Why this exists

Most RAG (Retrieval-Augmented Generation) systems use a simple "retrieve then synthesize" pipeline. This fails in two ways:
1. **Low Recall**: A single generic search query misses nuance.
2. **Hallucination**: The synthesizer drifts from the retrieved context.

ReMi solves this using a **multi-agent feedback loop**. If the initial report isn't sufficiently grounded in the retrieved sources, the system detects the failure, reformulates the search strategy based on the exact failure metrics, and tries again.

## Architecture

ReMi uses a pipeline of specialized agents:

1. **Planner**: Decomposes the user's query into focused sub-questions and generates HyDE (Hypothetical Document Embeddings) search variants for each.
2. **Researcher**: Retrieves chunks for each sub-question using a thread-safe `HybridStore` (FAISS dense + BM25 sparse + Reciprocal Rank Fusion) and synthesizes an initial answer.
3. **Analyzer**: Uses a fast inference model (Groq) to score and filter retrieved chunks. Low-confidence chunks are dropped; if all chunks fail the threshold, the best one is kept but flagged.
4. **Synthesizer**: Compiles the findings into a continuous Markdown report with `[^1]` footnote markers.
5. **Grounder**: Uses `difflib` fuzzy matching to link each footnote back to absolute character spans in the source document.
6. **Evaluator**: Scores the final report on 4 metrics: `citation_coverage`, `citation_utilization`, `answer_relevance`, and `hallucination_risk`.

### The Agentic Feedback Loop

This is the core of ReMi. If the Evaluator's scores fall below configured thresholds (e.g., `citation_coverage < 0.55`), the workflow triggers a retry.
Crucially, it does **not** just blindly retry or lower thresholds. It passes the exact failure scores back to the Planner, along with the previous failed sub-questions, instructing the LLM to reformulate its search strategy.

## Design Decisions & Known Limitations

- **Model Abstraction**: Model names are strictly abstracted behind `config.py`. No hardcoded strings. This protects against rapid model deprecation (e.g., Google's 1.5 Flash deprecation).
- **Concurrency**: `HybridStore` is fully thread-safe. Dense embedding calls happen *outside* the read/write lock so network latency doesn't serialize concurrent reads.
- **difflib Grounding Proxy**: The Grounder uses string similarity (0.6 threshold). This measures *traceability* (did the LLM use the text?), not *truth* (did the LLM use it correctly?). 
- **FAISS Deletion**: FAISS `IndexFlatIP` does not support in-place deletion. Deleting a document via the API removes it from the registry, but its chunks remain searchable until the index is rebuilt.

## Setup & Running

```bash
# 1. Install dependencies
git clone https://github.com/divya-3005/ReMi.git
cd ReMi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY and GROQ_API_KEY

# 3. Start the API server
python cli.py serve
# Or use uvicorn directly: uvicorn src.api.main:app --reload

# 4. Open the Frontend
# Open frontend/index.html in your browser.
# Ensure API_BASE points to http://localhost:8000
```

## CLI Usage

ReMi also includes a Typer CLI:

```bash
# Upload a document
python cli.py upload /path/to/document.pdf

# Ask a question
python cli.py ask "What were the main causes of the 2008 financial crisis?"

# Get raw JSON output
python cli.py ask "What is quantitative easing?" --json
```

## Project Structure

- `src/agent/` - The core LLM agents and workflow logic.
- `src/api/` - FastAPI REST endpoints.
- `src/evaluation/` - Quality scoring metrics.
- `src/genai/` - LLM client wrappers and pure prompt functions.
- `src/grounding/` - difflib citation matching.
- `src/ingestion/` - PDF/TXT parsing and chunking.
- `src/models/` - Pydantic schemas.
- `src/vectorstore/` - Hybrid FAISS/BM25 retrieval.
- `frontend/` - HTML/CSS/JS UI implementation.
- `tests/` - Comprehensive test suite (153 tests).
