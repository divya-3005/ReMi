# ResearchMind Document Ingestion & AI Agent Pipeline

A robust, local, evidence-grounded document ingestion and AI agent pipeline. It processes PDFs and plain text files, extracts NLP metadata, vectorizes the chunks using FAISS, and utilizes a multi-step autonomous AI agent powered by Google Gemini to research, ground, and synthesize rigorous evidence-backed reports.

## Features

### Core Data Pipeline
- **Loader & Cleaner**: Ingests `.pdf` (using `pypdf`) and `.txt` files. Removes excessive whitespace, repairs hyphenated words, and heuristically strips repeating headers/footers.
- **Chunker**: Splits text using a sliding window (500 tokens size, 50 tokens overlap) with `tiktoken` (`cl100k_base`).
- **NLP Engine**: Extracts keywords and named entities (using `spaCy` and `PyTextRank`), storing both per-chunk and document-level insights.
- **Vector Store**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to embed chunks locally and persists them in a FAISS index (`data/faiss_index.bin` and `data/store.json`).

### Generative AI & Agentic Layer
- **QA & Summarization**: Direct integration with Gemini 1.5 Flash (`google-generativeai`) to provide RAG-based question answering and hierarchical document summarization.
- **Autonomous Research Agent**: 
  - **Planner**: Decomposes high-level research questions into 3-5 sub-questions.
  - **Researcher**: Answers each sub-question against the vector store.
  - **Analyzer**: Ranks, deduplicates, and filters low-confidence findings.
  - **Synthesizer**: Constructs a comprehensive Markdown-formatted executive report.
- **Evidence Grounding**: Analyzes the generated report and explicitly links every generated sentence to the exact string span in the source documents using semantic sliding windows. It inserts Markdown footnotes, flags hallucinated statements with `[UNGROUNDED]`, and calculates `Faithfulness` and `Coverage` scores.

## Project Structure
```
researchmind/
├── ingestion/        # Document loaders, cleaners, and chunkers
├── nlp/              # spaCy extraction for entities and keywords
├── vectorstore/      # Local FAISS index and sentence-transformer embedder
├── genai/            # Gemini API integration, prompts, QA, and summarizer
├── agent/            # Autonomous research workflow (planner, researcher, etc.)
├── grounding/        # Evidence extractor, linker, scorer, and report generator
├── storage/          # Local JSON document store
├── models/           # Dataclasses for documents and agent states
├── tests/            # pytest unit tests
├── cli.py            # Rich Typer CLI
└── .env              # (Add your GEMINI_API_KEY here)
```

## Installation

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Google Gemini:
   Create a `.env` file in the root directory (you can copy `.env.example`) and add your API key:
   ```
   GEMINI_API_KEY=your_google_ai_key
   ```

4. Download the `spaCy` model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## CLI Usage

### Data Ingestion
- `python cli.py ingest <path_to_file_or_directory>`: Ingests documents, chunks them, and stores them in JSON.
- `python cli.py list`: Lists all ingested documents.
- `python cli.py show <doc_id>`: Shows details and chunks of a document.
- `python cli.py reset`: Deletes all stored documents and resets the databases.

### NLP Processing
- `python cli.py nlp`: Runs NLP extraction on all un-processed chunks to extract keywords and entities.

### Semantic Search & Q&A
- `python cli.py search "query"`: Performs a pure vector search across the FAISS database.
- `python cli.py summarize <doc_id>`: Synthesizes a hierarchical summary of a specific document.
- `python cli.py ask "query"`: Retrieves context and generates an exact answer, complete with citation mappings and grounding metrics.

### Autonomous Agent
- `python cli.py research "high-level question" --min-confidence 0.3`: Triggers the multi-step research agent to synthesize a fully cited Markdown report.
- `python cli.py reports`: Lists all previously saved agent research reports.

## Running Tests
Run tests using `pytest`:
```bash
pytest
```
