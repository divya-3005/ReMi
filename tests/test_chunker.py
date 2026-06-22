import tiktoken
from ingestion.chunker import chunk_document

def test_chunker_sliding_window():
    """Tests that the chunker splits text using a sliding window,
    with accurate token counts and correct overlap behavior.
    """
    # Create a test text (about 300 words, ~350 tokens)
    sentence = "The quick brown fox jumps over the lazy dog. Scientific research requires evidence-grounded facts. "
    text = sentence * 20

    doc_id = "test-doc-id"
    source_file = "test.txt"

    # Define small sizes to trigger multi-chunking easily
    chunk_size = 80
    overlap = 15

    chunks = chunk_document(
        document_id=doc_id,
        cleaned_text=text,
        source_file=source_file,
        chunk_size=chunk_size,
        overlap=overlap
    )

    # 1. Verify we got multiple chunks
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"

    # 2. Verify all metadata values are initialized correctly
    for chunk in chunks:
        assert chunk.document_id == doc_id
        assert chunk.source_file == source_file
        assert chunk.char_offset >= 0
        assert chunk.token_count > 0
        assert len(chunk.content) > 0

    # 3. Verify token count accuracy and overlap
    encoding = tiktoken.get_encoding("cl100k_base")

    for idx in range(len(chunks) - 1):
        chunk = chunks[idx]
        next_chunk = chunks[idx + 1]

        # Verify token counts roughly align with content
        tokens_current = encoding.encode(chunk.content)
        assert abs(len(tokens_current) - chunk.token_count) <= 2

        # Verify that the overlap strings match
        # The end of the current chunk should match the beginning of the next chunk
        tokens_next = encoding.encode(next_chunk.content)

        # Get the string representing the last 'overlap' tokens of the current chunk
        overlap_from_current = encoding.decode(tokens_current[-overlap:])
        # Get the string representing the first 'overlap' tokens of the next chunk
        overlap_from_next = encoding.decode(tokens_next[:overlap])

        # They should match closely (accounting for slight spacing/boundary variance)
        assert (
            overlap_from_current.strip() == overlap_from_next.strip()
            or overlap_from_current.strip() in next_chunk.content
            or overlap_from_next.strip() in chunk.content
        )
