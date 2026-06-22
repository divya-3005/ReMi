# ResearchMind Document Ingestion Pipeline

A robust, local, evidence-grounded document ingestion pipeline for processing PDFs and plain text files. It performs loader ingestion, heuristic cleaning (including header/footer stripping and broken hyphenation fixes), sliding-window token chunking, and metadata storage in a local JSON file.

## Features
- **Loader**: Ingests `.pdf` (using `pypdf`) and `.txt` files.
- **Cleaner**: Removes excessive whitespace, repairs hyphenated words split by line breaks, and heuristically strips repeating headers/footers (lines under 5 words that repeat across multiple pages).
- **Chunker**: Splits text using a sliding window (500 tokens size, 50 tokens overlap) with `tiktoken` (`cl100k_base`). It accurately maps chunks to their starting character offset and page numbers.
- **Store**: Saves documents and chunk details to a local `data/store.json` using standard JSON.
- **CLI**: Supports `ingest <path>`, `list`, and `show <doc_id>` commands using `typer` and `rich`.

## Project Structure
```
researchmind/
├── ingestion/
│   ├── __init__.py
│   ├── loader.py        # load PDF and .txt files
│   ├── cleaner.py       # clean raw extracted text
│   └── chunker.py       # split into overlapping chunks
├── storage/
│   ├── __init__.py
│   └── document_store.py  # simple JSON-based local store
├── models/
│   └── document.py      # Document and Chunk dataclasses
├── tests/
│   └── test_chunker.py  # pytest unit tests
├── cli.py               # CLI entrypoint
├── requirements.txt
└── README.md
```

## Installation

1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## CLI Usage

### Ingest a file or directory:
```bash
python cli.py ingest <path_to_file_or_directory>
```

### List all ingested documents:
```bash
python cli.py list
```

### Show details and chunks of a document:
```bash
python cli.py show <doc_id>
```

## Running Tests
Run tests using `pytest`:
```bash
pytest
```
