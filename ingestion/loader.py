import os
import datetime
from typing import Tuple, Dict, Any, List
import fitz  # PyMuPDF

def load_file(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """Loads a PDF or TXT file and returns its raw text (pages separated by form-feed \\x0c) and metadata.

    Args:
        file_path: Path to the file.

    Returns:
        A tuple of (raw_text, metadata_dict).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported or corrupt.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    ingestion_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

    metadata = {
        "filename": filename,
        "file_size_bytes": file_size,
        "ingested_at": ingestion_time,
        "page_count": None
    }

    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
            metadata["page_count"] = 1
            return raw_text, metadata
        except Exception as e:
            raise ValueError(f"Failed to read text file {filename}: {str(e)}")

    elif ext == ".pdf":
        try:
            doc = fitz.open(file_path)
            pages_text: List[str] = []
            for page in doc:
                text = page.get_text("text") or ""
                pages_text.append(text)

            raw_text = "\x0c".join(pages_text)
            metadata["page_count"] = len(doc)
            return raw_text, metadata
        except Exception as e:
            raise ValueError(f"Failed to read PDF file {filename}: {str(e)}")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
