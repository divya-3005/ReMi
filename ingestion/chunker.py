import uuid
import bisect
from typing import List
import tiktoken
from models.document import Chunk

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def split_text(self, text: str) -> List[str]:
        final_chunks = []
        separator = self.separators[-1]
        for s in self.separators:
            if s == "" or s in text:
                separator = s
                break
                
        if separator:
            splits = text.split(separator)
            if separator == ". ":
                splits = [s + ". " for s in splits[:-1]] + [splits[-1]]
        else:
            splits = list(text)
            
        good_splits = []
        for s in splits:
            if self._length(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                other_info = self.split_text(s)
                final_chunks.extend(other_info)
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)
            
        return [c.strip() for c in final_chunks if c.strip()]

    def _length(self, text: str) -> int:
        return len(self.encoding.encode(text))
        
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total = 0
        for d in splits:
            _len = self._length(d)
            if total + _len + (self._length(separator) if current_doc and separator != ". " else 0) > self.chunk_size:
                if total > 0:
                    doc = (separator if separator != ". " else "").join(current_doc)
                    docs.append(doc)
                    while total > self.chunk_overlap or (total + _len > self.chunk_size and total > 0):
                        total -= self._length(current_doc[0]) + (self._length(separator) if len(current_doc) > 1 and separator != ". " else 0)
                        current_doc.pop(0)
            current_doc.append(d)
            total += _len + (self._length(separator) if len(current_doc) > 1 and separator != ". " else 0)
        if current_doc:
            doc = (separator if separator != ". " else "").join(current_doc)
            docs.append(doc)
        return docs

def chunk_document(
    document_id: str,
    cleaned_text: str,
    source_file: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Chunk]:
    if not cleaned_text:
        return []

    # Track page start char offsets (0-indexed characters)
    page_starts = [0]
    pos = 0
    while True:
        pos = cleaned_text.find("\x0c", pos)
        if pos == -1:
            break
        page_starts.append(pos + 1)
        pos += 1

    # Split using semantic chunker
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    text_chunks = splitter.split_text(cleaned_text)

    chunks: List[Chunk] = []
    last_char_idx = 0

    for text_chunk in text_chunks:
        # Find exactly where this chunk started in the original cleaned_text
        start_char = cleaned_text.find(text_chunk, last_char_idx)
        if start_char == -1:
            # Fallback if somehow not found (shouldn't happen with exact match)
            start_char = last_char_idx
            
        page_idx = bisect.bisect_right(page_starts, start_char)
        page_number = page_idx if page_idx <= len(page_starts) else len(page_starts)

        token_count = splitter._length(text_chunk)

        chunk_obj = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            content=text_chunk,
            source_file=source_file,
            page_number=page_number,
            char_offset=start_char,
            token_count=token_count
        )
        chunks.append(chunk_obj)
        
        last_char_idx = start_char + len(text_chunk) - 50 # slight rollback for overlap

    return chunks
