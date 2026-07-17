"""
src/ingestion/errors.py
───────────────────────
Named exceptions for the ingestion layer.

Using specific exception types instead of bare ValueError/RuntimeError means:
- The API layer can catch each type and return a structured JSON error response
  with an appropriate HTTP status code and a user-facing suggestion.
- Tests can assert the exact failure mode rather than just "some error happened".
- Silent failures (especially empty-text PDFs) surface immediately instead of
  propagating as mysteriously empty strings.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for all document ingestion errors."""


class UnsupportedFileTypeError(IngestionError):
    """
    Raised when a file extension is not supported.

    Supported: .pdf, .txt
    Common unsupported: .docx, .pptx, .xlsx, .html
    """


class EncryptedFileError(IngestionError):
    """
    Raised when a PDF requires a password to open.

    Suggestion: Decrypt the PDF before uploading, or use an unprotected export.
    """


class EmptyTextError(IngestionError):
    """
    Raised when a PDF contains no extractable text layer.

    Most common cause: scanned/image-only PDFs where the content is pixel data,
    not a text layer. OCR is not supported in this version.

    Known limitation documented in README.
    """


class CorruptedFileError(IngestionError):
    """
    Raised when PyMuPDF fails to open a file due to data corruption.
    """
