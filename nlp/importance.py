import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import spacy

_nlp_model = None

def get_nlp() -> spacy.Language:
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = spacy.load("en_core_web_sm")
    return _nlp_model

def score_sentences(text: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Scores sentences in a text using a TextRank-style approach.
    
    Args:
        text: The input text (e.g. a chunk).
        top_n: Number of top sentences to return.
        
    Returns:
        List of dicts with 'sentence' and 'score', sorted by score descending.
    """
    if not text.strip():
        return []
        
    nlp = get_nlp()
    doc = nlp(text)
    
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if not sentences:
        return []
        
    if len(sentences) == 1:
        return [{"sentence": sentences[0], "score": 1.0}]
        
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # Happens if sentences contain only stop words or unsupported tokens
        return [{"sentence": s, "score": 1.0 / len(sentences)} for s in sentences[:top_n]]
        
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Build graph from similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    try:
        scores = nx.pagerank(nx_graph)
    except Exception:
        # Fallback if pagerank fails to converge
        scores = {i: 1.0 / len(sentences) for i in range(len(sentences))}
        
    ranked_sentences = []
    for i, s in enumerate(sentences):
        ranked_sentences.append({
            "sentence": s,
            "score": float(scores.get(i, 0.0))
        })
        
    ranked_sentences.sort(key=lambda x: x["score"], reverse=True)
    return ranked_sentences[:top_n]
