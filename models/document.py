from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class Document:
    """Represents an ingested document containing raw extracted text and metadata."""
    id: str
    filename: str
    raw_text: str
    file_size_bytes: int
    ingested_at: str  # ISO 8601 string
    page_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    nlp: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    """Represents a clean tokenized chunk of a document."""
    chunk_id: str
    document_id: str
    content: str
    source_file: str
    page_number: Optional[int]  # 1-indexed best guess
    char_offset: int
    token_count: int
