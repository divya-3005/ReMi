"""
src/ingestion/loader.py
───────────────────────
Document loading and parsing.

Uses PyMuPDF (fitz) for PDF extraction. Handles text-layer extraction natively,
but explicitly rejects scanned/image-only PDFs (EmptyTextError) and encrypted
PDFs (EncryptedFileError).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import fitz  # PyMuPDF

from src.ingestion.errors import (
    CorruptedFileError,
    EmptyTextError,
    EncryptedFileError,
    UnsupportedFileTypeError,
)
from src.models.schemas import DocumentMetadata


class DocumentLoader:
    """Loads and parses PDF and TXT files."""

    def load(self, file_path: str | Path) -> Tuple[DocumentMetadata, str]:
        """
        Load a file and extract its text and metadata.

        Args:
            file_path: Path to the file to load.

        Returns:
            Tuple containing DocumentMetadata and the extracted raw text string.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnsupportedFileTypeError: If the extension is not .pdf or .txt.
            CorruptedFileError: If PyMuPDF cannot parse the PDF.
            EncryptedFileError: If the PDF requires a password.
            EmptyTextError: If the PDF has no text layer (e.g. scanned).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._load_pdf(path)
        elif ext == ".txt":
            return self._load_txt(path)
        else:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {ext}. "
                "Only .pdf and .txt are supported."
            )

    def _load_txt(self, path: Path) -> Tuple[DocumentMetadata, str]:
        """Load a plain text file, falling back to Latin-1 if UTF-8 fails."""
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        meta = DocumentMetadata(
            filename=path.name,
            file_type="txt",
            num_pages=1,
        )
        return meta, text

    def _load_pdf(self, path: Path) -> Tuple[DocumentMetadata, str]:
        """
        Load a PDF using PyMuPDF (fitz).
        Extracts text block-by-block to preserve layout.
        """
        try:
            doc = fitz.open(str(path))
        except fitz.FileDataError as e:
            raise CorruptedFileError(f"Failed to open PDF due to corruption: {e}")

        if doc.needs_pass:
            raise EncryptedFileError(
                f"Password-protected PDF; decryption not supported: {path.name}"
            )

        num_pages = len(doc)
        text_parts = []

        for page in doc:
            # get_text("blocks") returns a list of blocks, each a tuple:
            # (x0, y0, x1, y1, "lines in block", block_no, block_type)
            # block_type 0 is text, 1 is image.
            blocks = page.get_text("blocks")
            
            # Sort blocks by vertical position (y0), then horizontal (x0)
            # This is a naive heuristic for reading order in multi-column layouts.
            # Documented limitation: it will struggle on complex layouts.
            blocks.sort(key=lambda b: (b[1], b[0]))
            
            for block in blocks:
                if block[6] == 0:  # text block
                    text_parts.append(block[4].strip())
            
            text_parts.append("\n\n") # Page break separator

        extracted_text = "\n".join(text_parts).strip()

        if not extracted_text:
            raise EmptyTextError(
                "No text layer found. This may be a scanned PDF. "
                "OCR is not supported in this version."
            )

        meta = DocumentMetadata(
            filename=path.name,
            file_type="pdf",
            num_pages=num_pages,
        )
        
        return meta, extracted_text
