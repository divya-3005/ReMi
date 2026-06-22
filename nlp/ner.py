import spacy
from typing import List, Dict, Any
from collections import defaultdict

# Cache the model to avoid reloading for every text
_nlp_model = None

def get_nlp() -> spacy.Language:
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp_model = spacy.load("en_core_web_sm")
    return _nlp_model

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extracts entities from the given text using spaCy.
    Filters to: PERSON, ORG, GPE, DATE, EVENT, PRODUCT, LAW.
    
    Args:
        text: Input string.
        
    Returns:
        List of dicts with 'text', 'label', and 'count', sorted by count descending.
    """
    if not text.strip():
        return []
        
    nlp = get_nlp()
    doc = nlp(text)
    
    target_labels = {"PERSON", "ORG", "GPE", "DATE", "EVENT", "PRODUCT", "LAW"}
    
    # Use (text, label) as key for deduplication
    entity_counts = defaultdict(int)
    
    for ent in doc.ents:
        if ent.label_ in target_labels:
            clean_text = ent.text.strip()
            if clean_text:
                entity_counts[(clean_text, ent.label_)] += 1
                
    result = []
    for (ent_text, label), count in entity_counts.items():
        result.append({
            "text": ent_text,
            "label": label,
            "count": count
        })
        
    # Sort by frequency descending
    result.sort(key=lambda x: x["count"], reverse=True)
    return result

def aggregate_entities(chunk_entities_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Aggregates per-chunk entities to a document-level summary.
    
    Args:
        chunk_entities_list: A list where each element is the return value of extract_entities.
        
    Returns:
        A list of aggregated entity dicts, sorted by count descending.
    """
    entity_counts = defaultdict(int)
    for chunk_ents in chunk_entities_list:
        for ent in chunk_ents:
            entity_counts[(ent["text"], ent["label"])] += ent["count"]
            
    result = []
    for (ent_text, label), count in entity_counts.items():
        result.append({
            "text": ent_text,
            "label": label,
            "count": count
        })
        
    result.sort(key=lambda x: x["count"], reverse=True)
    return result
