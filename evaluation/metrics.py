import numpy as np
from typing import List
from models.research import LinkedAnswer
from grounding.scorer import faithfulness_score as original_faithfulness_score
from vectorstore.embedder import embed_query, embed_texts

def faithfulness(linked_answer: LinkedAnswer) -> float:
    return original_faithfulness_score(linked_answer)

def hallucination_risk(linked_answer: LinkedAnswer) -> float:
    return 1.0 - faithfulness(linked_answer)

def answer_relevance(question: str, answer: str) -> float:
    if not answer.strip():
        return 0.0
    q_emb = embed_query(question)
    a_emb = embed_query(answer)
    
    q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
    a_norm = np.linalg.norm(a_emb, axis=1, keepdims=True)
    
    q_norm[q_norm == 0] = 1e-10
    a_norm[a_norm == 0] = 1e-10
    
    q_emb_norm = q_emb / q_norm
    a_emb_norm = a_emb / a_norm
    
    sim = float(np.dot(q_emb_norm, a_emb_norm.T)[0][0])
    return max(0.0, min(1.0, sim))

def context_precision(question: str, chunks: List[str]) -> float:
    if not chunks:
        return 0.0
        
    q_emb = embed_query(question)
    chunks_emb = embed_texts(chunks)
    
    q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
    chunks_norm = np.linalg.norm(chunks_emb, axis=1, keepdims=True)
    
    q_norm[q_norm == 0] = 1e-10
    chunks_norm[chunks_norm == 0] = 1e-10
    
    q_emb_norm = q_emb / q_norm
    chunks_emb_norm = chunks_emb / chunks_norm
    
    similarities = np.dot(chunks_emb_norm, q_emb_norm.T).flatten()
    
    relevant_count = sum(1 for sim in similarities if sim > 0.4)
    return float(relevant_count / len(chunks))
