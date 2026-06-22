import uuid
import bisect
from typing import List
import tiktoken
from models.document import Chunk

def chunk_document(
    document_id: str,
    cleaned_text: str,
    source_file: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Chunk]:
    """Splits cleaned text into overlapping chunks using tiktoken.

    Args:
        document_id: The ID of the parent document.
        cleaned_text: The cleaned text to chunk (contains \\x0c as page separators).
        source_file: The name of the source file.
        chunk_size: Target token size per chunk.
        overlap: Token overlap between adjacent chunks.

    Returns:
        A list of Chunk objects.
    """
    if not cleaned_text:
        return []

    # Initialize tokenizer
    encoding = tiktoken.get_encoding("cl100k_base")

    # Encode the text to tokens
    tokens = encoding.encode(cleaned_text)

    # 1. Track page start char offsets
    # The first page starts at char index 0
    page_starts = [0]
    pos = 0
    while True:
        pos = cleaned_text.find("\x0c", pos)
        if pos == -1:
            break
        page_starts.append(pos + 1)
        pos += 1

    # 2. Build token-to-byte mapping to handle multi-byte Unicode characters accurately
    raw_bytes = cleaned_text.encode("utf-8")
    token_byte_offsets: List[int] = []
    current_byte_offset = 0
    for t in tokens:
        token_byte_offsets.append(current_byte_offset)
        # Length of raw bytes of this single token
        token_bytes = encoding.decode_single_token_bytes(t)
        current_byte_offset += len(token_bytes)

    # 3. Build character-to-byte starting position mapping
    char_to_byte: List[int] = []
    current_byte = 0
    for char in cleaned_text:
        char_to_byte.append(current_byte)
        current_byte += len(char.encode("utf-8"))
    # Append the total byte length to handle end-of-string offset mapping
    char_to_byte.append(current_byte)

    # Helper function to convert a byte offset to character index
    def get_char_index(byte_offset: int) -> int:
        # bisect_right returns insertion point. Subtract 1 to get character index
        idx = bisect.bisect_right(char_to_byte, byte_offset)
        return max(0, idx - 1)

    # 4. Chunking loop using sliding window
    chunks: List[Chunk] = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size  # Prevent infinite loop if overlap >= chunk_size

    i = 0
    while i < len(tokens):
        # Determine the window slice
        window_tokens = tokens[i : i + chunk_size]
        if not window_tokens:
            break

        # Start and end byte offsets of current token window
        start_byte_offset = token_byte_offsets[i]
        
        # End byte offset computation
        if i + chunk_size < len(tokens):
            end_byte_offset = token_byte_offsets[i + chunk_size]
        else:
            end_byte_offset = len(raw_bytes)

        # Map byte offsets back to clean character indexes
        start_char = get_char_index(start_byte_offset)
        end_char = get_char_index(end_byte_offset)

        # Slice the content
        content = cleaned_text[start_char:end_char]

        # Guess page number based on where the chunk starts
        # bisect_right on page_starts tells us the page (1-indexed)
        page_idx = bisect.bisect_right(page_starts, start_char)
        page_number = page_idx if page_idx <= len(page_starts) else len(page_starts)

        # Add chunk
        chunk_obj = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            content=content,
            source_file=source_file,
            page_number=page_number,
            char_offset=start_char,
            token_count=len(window_tokens)
        )
        chunks.append(chunk_obj)

        # If this chunk reaches the end of the document, stop
        if i + chunk_size >= len(tokens):
            break

        i += step

    return chunks
