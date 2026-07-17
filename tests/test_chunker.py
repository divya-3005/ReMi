"""
tests/test_chunker.py
─────────────────────
Phase 3 test suite: RecursiveChunker.

Tests verify hierarchical splitting, overlap logic, edge cases (empty strings),
and the exact correctness of char_start / char_end offsets which are critical
for the grounding layer.
"""

from __future__ import annotations

from src.ingestion.chunker import RecursiveChunker
from src.models.schemas import DocumentMetadata


def test_chunker_initializes_with_settings(mock_settings):
    chunker = RecursiveChunker(mock_settings)
    assert chunker.chunk_size == mock_settings.chunk_size
    assert chunker.chunk_overlap == mock_settings.chunk_overlap


def test_empty_string_returns_empty_list(mock_settings):
    chunker = RecursiveChunker(mock_settings)
    meta = DocumentMetadata(filename="empty.txt", file_type="txt", num_pages=1)
    chunks = chunker.chunk("", meta)
    assert len(chunks) == 0


def test_short_string_returns_single_chunk(mock_settings, short_text):
    # chunk_size is 500 in mock_settings, short_text is ~60 chars
    chunker = RecursiveChunker(mock_settings)
    meta = DocumentMetadata(filename="short.txt", file_type="txt", num_pages=1)
    chunks = chunker.chunk(short_text, meta)
    
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == short_text
    assert chunk.char_start == 0
    assert chunk.char_end == len(short_text)
    assert chunk.page_number == 1
    assert chunk.doc_id == meta.doc_id


def test_long_string_is_split_correctly(mock_settings, sample_text):
    # sample_text is ~600 chars. mock_settings has chunk_size=500, overlap=100
    chunker = RecursiveChunker(mock_settings)
    meta = DocumentMetadata(filename="long.txt", file_type="txt", num_pages=1)
    chunks = chunker.chunk(sample_text, meta)
    
    assert len(chunks) > 1
    
    # Verify no chunk exceeds chunk_size + generous margin for indivisible tokens
    for chunk in chunks:
        assert len(chunk.text) <= chunker.chunk_size + 50


def test_char_offsets_are_exact(mock_settings, sample_text):
    chunker = RecursiveChunker(mock_settings)
    meta = DocumentMetadata(filename="test.txt", file_type="txt", num_pages=1)
    chunks = chunker.chunk(sample_text, meta)
    
    for chunk in chunks:
        # The substring in the original document MUST exactly match the chunk text
        extracted_from_source = sample_text[chunk.char_start:chunk.char_end]
        assert extracted_from_source == chunk.text


def test_overlap_is_respected(mock_settings, sample_text):
    # Force a small chunk size to ensure multiple splits
    mock_settings.__dict__['chunk_size'] = 100
    mock_settings.__dict__['chunk_overlap'] = 20
    
    chunker = RecursiveChunker(mock_settings)
    meta = DocumentMetadata(filename="test.txt", file_type="txt", num_pages=1)
    chunks = chunker.chunk(sample_text, meta)
    
    assert len(chunks) >= 2
    
    # Check overlap between chunk 0 and chunk 1
    c0 = chunks[0]
    c1 = chunks[1]
    
    # If c1 starts before c0 ends, there is overlap
    assert c1.char_start < c0.char_end
    
    # The actual overlap amount should be roughly the target overlap,
    # though it varies based on separator boundaries.
    overlap_chars = c0.char_end - c1.char_start
    assert overlap_chars > 0


def test_page_number_estimation(mock_settings):
    # A 3000 character document that is 3 pages long.
    # Page 1: 0-1000
    # Page 2: 1000-2000
    # Page 3: 2000-3000
    doc_text = "A" * 3000
    meta = DocumentMetadata(filename="doc.pdf", file_type="pdf", num_pages=3)
    
    mock_settings.__dict__['chunk_size'] = 100
    mock_settings.__dict__['chunk_overlap'] = 0
    
    chunker = RecursiveChunker(mock_settings)
    chunks = chunker.chunk(doc_text, meta)
    
    # First chunk should be page 1
    assert chunks[0].page_number == 1
    
    # A chunk starting at char 1500 should be page 2
    mid_chunk = next(c for c in chunks if c.char_start >= 1500)
    assert mid_chunk.page_number == 2
    
    # The last chunk should be page 3
    assert chunks[-1].page_number == 3
