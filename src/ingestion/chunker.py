"""
src/ingestion/chunker.py
────────────────────────
Recursive character text splitting with absolute offset tracking.

This chunker mimics LangChain's RecursiveCharacterTextSplitter but adds
mandatory tracking of char_start and char_end relative to the original
document. This is the foundation of the citation/grounding layer.
"""

from __future__ import annotations

from typing import List

from src.config import Settings
from src.models.schemas import Chunk, DocumentMetadata


class RecursiveChunker:
    """
    Splits text hierarchically using a predefined sequence of separators.
    Preserves exact character offsets from the original text.
    """

    # Try breaking on double newline (paragraphs), then single newline (lines),
    # then sentence boundaries, then spaces, then individual characters.
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, settings: Settings):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    def chunk(self, text: str, metadata: DocumentMetadata) -> List[Chunk]:
        """
        Split text into overlapping Chunks.
        
        Args:
            text: The full raw text of the document.
            metadata: DocumentMetadata to populate foreign keys and page estimates.
            
        Returns:
            List of Chunk objects.
        """
        if not text:
            return []

        # We need to track the absolute start offset of the current block
        # being processed. To do this recursively, we pass the current global
        # offset down into the helper.
        splits = self._split_text(text, self.SEPARATORS, global_offset=0)
        
        # Merge small adjacent splits into target chunk sizes
        merged_chunks = self._merge_splits(splits, text)
        
        # Estimate chars per page to calculate approximate page numbers
        chars_per_page = max(1, len(text) // max(1, metadata.num_pages))
        
        result = []
        for start_idx, end_idx, chunk_text in merged_chunks:
            page_estimate = min(
                metadata.num_pages,
                max(1, (start_idx // chars_per_page) + 1)
            )
            
            result.append(
                Chunk(
                    doc_id=metadata.doc_id,
                    text=chunk_text,
                    page_number=page_estimate,
                    char_start=start_idx,
                    char_end=end_idx
                )
            )
            
        return result

    def _split_text(self, text: str, separators: List[str], global_offset: int) -> List[tuple[int, int, str]]:
        """
        Recursively split text.
        Returns a list of tuples: (global_start, global_end, subtext).
        """
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [(global_offset, global_offset + len(text), text)]

        # Find the first separator that actually exists in the text
        separator = ""
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # Fallback to character splitting
        if separator == "":
            return [
                (global_offset + i, global_offset + i + 1, text[i])
                for i in range(len(text))
            ]

        # Split by the chosen separator
        # We use re.split to keep the separator if needed, but simple string
        # split is faster. We have to preserve the separator to not lose chars.
        
        # NOTE: Hierarchical splitting degrades on tables, code blocks, and bullet lists.
        # Table rows may be split mid-cell, producing semantically invalid chunks.
        # This is a known limitation; a future version could detect and preserve table structures.
        
        parts = text.split(separator)
        
        splits = []
        current_offset = global_offset
        
        for i, part in enumerate(parts):
            is_last = (i == len(parts) - 1)
            part_with_sep = part if is_last else part + separator
            
            if len(part_with_sep) > self.chunk_size and len(separators) > 1:
                # Part is still too big; recurse with the next separators
                next_seps = separators[separators.index(separator) + 1:]
                sub_splits = self._split_text(part_with_sep, next_seps, current_offset)
                splits.extend(sub_splits)
            else:
                splits.append(
                    (current_offset, current_offset + len(part_with_sep), part_with_sep)
                )
                
            current_offset += len(part_with_sep)
            
        return splits

    def _merge_splits(self, splits: List[tuple[int, int, str]], original_text: str) -> List[tuple[int, int, str]]:
        """
        Merge small consecutive splits into chunks approaching chunk_size,
        applying chunk_overlap when rolling over to the next chunk.
        """
        if not splits:
            return []

        merged = []
        current_doc: List[tuple[int, int, str]] = []
        current_len = 0
        current_start = -1

        for split_start, split_end, split_text in splits:
            if current_start == -1:
                current_start = split_start
                
            split_len = len(split_text)
            
            if current_len + split_len > self.chunk_size and current_len > 0:
                # Finalize the current chunk
                end_idx = current_start + current_len
                chunk_text = original_text[current_start:end_idx]
                merged.append((current_start, end_idx, chunk_text))
                
                # Setup the next chunk to include overlap from the end of the previous
                # Walk backward through current_doc until we have approximately 'overlap' chars
                overlap_len = 0
                overlap_docs: List[tuple[int, int, str]] = []
                for doc_start, doc_end, doc_text in reversed(current_doc):
                    if overlap_len + len(doc_text) > self.chunk_overlap and overlap_len > 0:
                        break
                    overlap_docs.insert(0, (doc_start, doc_end, doc_text))
                    overlap_len += len(doc_text)
                
                current_doc = overlap_docs
                current_doc.append((split_start, split_end, split_text))
                current_start = overlap_docs[0][0] if overlap_docs else split_start
                current_len = overlap_len + split_len
                
            else:
                current_doc.append((split_start, split_end, split_text))
                current_len += split_len

        # Add the final chunk if anything remains
        if current_doc:
            end_idx = current_start + current_len
            chunk_text = original_text[current_start:end_idx]
            merged.append((current_start, end_idx, chunk_text))

        return merged
