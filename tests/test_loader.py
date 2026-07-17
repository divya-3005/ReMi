"""
tests/test_loader.py
────────────────────
Phase 2 test suite: DocumentLoader.

All PDF-related tests use mocked fitz objects — no real PDF files needed
for unit tests (real files are used only in integration tests).
The .txt tests use real temp files via pytest's tmp_path fixture.

Zero external API calls. Zero real PDF dependencies at test time.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.ingestion.loader import DocumentLoader
from src.ingestion.errors import (
    CorruptedFileError,
    EmptyTextError,
    EncryptedFileError,
    UnsupportedFileTypeError,
)
from src.models.schemas import DocumentMetadata


SAMPLE_TXT_CONTENT = """\
Line one of the document.
Line two with more content.

Paragraph break here.
Final line.
"""


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def loader() -> DocumentLoader:
    return DocumentLoader()


@pytest.fixture
def txt_file(tmp_path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text(SAMPLE_TXT_CONTENT, encoding="utf-8")
    return p


@pytest.fixture
def latin1_txt_file(tmp_path) -> Path:
    """A .txt file encoded in Latin-1 (not UTF-8)."""
    p = tmp_path / "latin1.txt"
    p.write_bytes("Café résumé naïve".encode("latin-1"))
    return p


@pytest.fixture
def docx_file(tmp_path) -> Path:
    p = tmp_path / "doc.docx"
    p.write_bytes(b"PK\x03\x04")  # fake zip/docx bytes
    return p


# ── TXT loading ───────────────────────────────────────────────────────────────

class TestLoadTxt:
    def test_returns_metadata_and_text(self, loader, txt_file):
        meta, text = loader.load(str(txt_file))
        assert isinstance(meta, DocumentMetadata)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_metadata_filename_correct(self, loader, txt_file):
        meta, _ = loader.load(str(txt_file))
        assert meta.filename == "sample.txt"

    def test_metadata_file_type_is_txt(self, loader, txt_file):
        meta, _ = loader.load(str(txt_file))
        assert meta.file_type == "txt"

    def test_metadata_num_pages_is_one_for_txt(self, loader, txt_file):
        meta, _ = loader.load(str(txt_file))
        assert meta.num_pages == 1

    def test_text_contains_expected_content(self, loader, txt_file):
        _, text = loader.load(str(txt_file))
        assert "Line one" in text
        assert "Paragraph break" in text

    def test_doc_id_is_valid_uuid(self, loader, txt_file):
        meta, _ = loader.load(str(txt_file))
        assert len(meta.doc_id) == 36
        assert meta.doc_id.count("-") == 4

    def test_latin1_file_loaded_without_error(self, loader, latin1_txt_file):
        meta, text = loader.load(str(latin1_txt_file))
        assert len(text) > 0
        assert "Caf" in text  # "Café" decoded correctly

    def test_nonexistent_file_raises_file_not_found(self, loader, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader.load(str(tmp_path / "ghost.txt"))

    def test_unsupported_extension_raises_error(self, loader, docx_file):
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            loader.load(str(docx_file))
        assert "docx" in str(exc_info.value).lower()


# ── PDF loading (all fitz mocked) ────────────────────────────────────────────

class TestLoadPdf:
    def _make_mock_fitz_doc(
        self,
        num_pages: int = 3,
        text_per_page: str = "Some extracted text from page.",
        needs_pass: bool = False,
        raises_on_open: bool = False,
    ) -> MagicMock:
        """Build a mock fitz.Document with configurable behaviour."""
        mock_doc = MagicMock()
        mock_doc.needs_pass = needs_pass
        mock_doc.__len__ = MagicMock(return_value=num_pages)

        # Each page returns a list of text blocks; we simulate one block per page
        def make_page(text):
            page = MagicMock()
            page.get_text.return_value = [
                (0, 0, 100, 20, text, 0, 0)  # (x0, y0, x1, y1, text, ?, ?)
            ]
            return page

        mock_doc.__iter__ = MagicMock(
            return_value=iter([make_page(text_per_page)] * num_pages)
        )
        return mock_doc

    def test_valid_pdf_returns_metadata_and_text(self, loader, tmp_path):
        pdf_path = str(tmp_path / "paper.pdf")
        Path(pdf_path).touch()

        mock_doc = self._make_mock_fitz_doc(num_pages=5, text_per_page="Research findings.")
        with patch("fitz.open", return_value=mock_doc):
            meta, text = loader.load(pdf_path)

        assert meta.file_type == "pdf"
        assert meta.num_pages == 5
        assert "Research findings" in text

    def test_pdf_metadata_filename_correct(self, loader, tmp_path):
        pdf_path = str(tmp_path / "report.pdf")
        Path(pdf_path).touch()

        mock_doc = self._make_mock_fitz_doc()
        with patch("fitz.open", return_value=mock_doc):
            meta, _ = loader.load(pdf_path)

        assert meta.filename == "report.pdf"

    def test_encrypted_pdf_raises_encrypted_error(self, loader, tmp_path):
        pdf_path = str(tmp_path / "secret.pdf")
        Path(pdf_path).touch()

        mock_doc = self._make_mock_fitz_doc(needs_pass=True)
        with patch("fitz.open", return_value=mock_doc):
            with pytest.raises(EncryptedFileError) as exc_info:
                loader.load(pdf_path)
        assert "password" in str(exc_info.value).lower()

    def test_scanned_pdf_raises_empty_text_error(self, loader, tmp_path):
        """A scanned/image-only PDF produces no text layer."""
        pdf_path = str(tmp_path / "scanned.pdf")
        Path(pdf_path).touch()

        mock_doc = self._make_mock_fitz_doc(num_pages=4, text_per_page="")
        with patch("fitz.open", return_value=mock_doc):
            with pytest.raises(EmptyTextError) as exc_info:
                loader.load(pdf_path)
        # Error message should mention OCR so the user understands why
        assert "ocr" in str(exc_info.value).lower() or "scanned" in str(exc_info.value).lower()

    def test_corrupted_pdf_raises_corrupted_error(self, loader, tmp_path):
        pdf_path = str(tmp_path / "corrupt.pdf")
        Path(pdf_path).touch()

        import fitz as _fitz
        with patch("fitz.open", side_effect=_fitz.FileDataError("bad data")):
            with pytest.raises(CorruptedFileError):
                loader.load(pdf_path)

    def test_pdf_text_is_non_empty_string(self, loader, tmp_path):
        pdf_path = str(tmp_path / "doc.pdf")
        Path(pdf_path).touch()

        mock_doc = self._make_mock_fitz_doc(num_pages=2, text_per_page="Important content here.")
        with patch("fitz.open", return_value=mock_doc):
            _, text = loader.load(pdf_path)

        assert isinstance(text, str)
        assert len(text.strip()) > 0
