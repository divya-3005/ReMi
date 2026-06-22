from models.research import LinkedAnswer
import numpy as np
from vectorstore.embedder import embed_query, embed_texts

def faithfulness_score(linked: LinkedAnswer) -> float:
    if not linked.sentences:
        return 0.0
    grounded_count = sum(1 for s in linked.sentences if s.grounded)
    return float(grounded_count / len(linked.sentences))

def coverage_score(question: str, linked: LinkedAnswer) -> float:
    grounded_spans = []
    for s in linked.sentences:
        if s.grounded and s.evidence:
            grounded_spans.append(s.evidence[0].span_text)
            
    if not grounded_spans:
        return 0.0
        
    q_emb = embed_query(question)
    spans_emb = embed_texts(grounded_spans)
    
    q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
    spans_norm = np.linalg.norm(spans_emb, axis=1, keepdims=True)
    
    q_norm[q_norm == 0] = 1e-10
    spans_norm[spans_norm == 0] = 1e-10
    
    q_emb_norm = q_emb / q_norm
    spans_emb_norm = spans_emb / spans_norm
    
    # q_emb is (1, dim), spans_emb is (N, dim)
    similarities = np.dot(spans_emb_norm, q_emb_norm.T)
    
    return float(np.mean(similarities))
