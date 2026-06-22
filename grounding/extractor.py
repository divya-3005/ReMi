import re
import numpy as np
from typing import List
from models.research import EvidenceSpan
from vectorstore.retriever import SearchResult
from vectorstore.embedder import embed_texts

def split_into_sentences(text: str) -> List[str]:
    # Simple regex to split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def extract_spans(answer: str, chunks: List[SearchResult], window_size: int = 3) -> List[EvidenceSpan]:
    """
    Extract exact evidence spans from chunks for each sentence in the answer.
    Returns one top-1 EvidenceSpan per sentence in the answer.
    """
    answer_sentences = split_into_sentences(answer)
    if not answer_sentences or not chunks:
        return []
        
    ans_embeddings = embed_texts(answer_sentences)
    
    # Pre-process chunks into sliding windows
    all_windows = [] # list of (chunk_idx, start_char, end_char, window_text, source_file)
    for chunk in chunks:
        chunk_text = chunk.chunk_text
        # Find sentence boundaries with character offsets
        matches = list(re.finditer(r'[^.!?]+[.!?]*', chunk_text))
        sentences_info = []
        for m in matches:
            text = m.group(0).strip()
            if text:
                sentences_info.append({
                    "text": text,
                    "start": m.start(),
                    "end": m.end()
                })
        
        # Build sliding windows
        if not sentences_info:
            continue
            
        for i in range(len(sentences_info)):
            window = sentences_info[i:i+window_size]
            window_text = " ".join([s["text"] for s in window])
            start_char = window[0]["start"]
            end_char = window[-1]["end"]
            all_windows.append((chunk.chunk_index, start_char, end_char, window_text, chunk.source_file))
            
    if not all_windows:
        # If chunks couldn't be parsed into sentences, return empty spans
        # This prevents crashing but will result in ungrounded sentences
        return [
            EvidenceSpan(-1, "", 0, 0, "", 0.0) for _ in answer_sentences
        ]
        
    window_texts = [w[3] for w in all_windows]
    window_embeddings = embed_texts(window_texts)
    
    # Calculate cosine similarity
    ans_norms = np.linalg.norm(ans_embeddings, axis=1, keepdims=True)
    win_norms = np.linalg.norm(window_embeddings, axis=1, keepdims=True)
    
    ans_norms[ans_norms == 0] = 1e-10
    win_norms[win_norms == 0] = 1e-10
    
    ans_embeddings_norm = ans_embeddings / ans_norms
    window_embeddings_norm = window_embeddings / win_norms
    
    similarities = np.dot(ans_embeddings_norm, window_embeddings_norm.T)
    
    results = []
    for i, _ in enumerate(answer_sentences):
        best_idx = np.argmax(similarities[i])
        best_score = similarities[i][best_idx]
        
        chunk_idx, start_char, end_char, window_text, source_file = all_windows[best_idx]
        
        span = EvidenceSpan(
            chunk_index=chunk_idx,
            source_file=source_file,
            start_char=start_char,
            end_char=end_char,
            span_text=window_text,
            relevance_score=float(best_score)
        )
        results.append(span)
        
    return results
