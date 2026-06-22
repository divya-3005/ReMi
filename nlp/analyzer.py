from typing import Dict, Any
from storage.document_store import DocumentStore
from nlp.keywords import extract_keywords
from nlp.ner import extract_entities, aggregate_entities
from nlp.importance import score_sentences

def analyze_document(doc_id: str, store: DocumentStore) -> bool:
    """
    Runs NLP analysis (keywords, entities, sentence importance) on a document
    and saves the results back into the store.
    
    Args:
        doc_id: The UUID of the document.
        store: The DocumentStore instance.
        
    Returns:
        True if successful, False if document was not found.
    """
    res = store.get_by_id(doc_id)
    if not res:
        return False
        
    doc, chunks = res
    if not chunks:
        doc.nlp = {
            "keywords": {"document_level": [], "per_chunk": {}},
            "entities": {"document_level": [], "per_chunk": {}},
            "importance": {"per_chunk": {}}
        }
        store.save(doc, chunks)
        return True
        
    # Prepare data for keyword extraction
    chunks_data = [{"chunk_id": c.chunk_id, "content": c.content} for c in chunks]
    keywords_result = extract_keywords(chunks_data)
    
    # Process entities and importance per chunk
    entities_per_chunk = {}
    chunk_entities_list = []
    
    importance_per_chunk = {}
    
    for chunk in chunks:
        # Entities
        ents = extract_entities(chunk.content)
        entities_per_chunk[chunk.chunk_id] = ents
        chunk_entities_list.append(ents)
        
        # Importance
        sents = score_sentences(chunk.content)
        importance_per_chunk[chunk.chunk_id] = sents
        
    # Aggregate entities
    doc_entities = aggregate_entities(chunk_entities_list)
    
    # Combine results
    doc.nlp = {
        "keywords": keywords_result,
        "entities": {
            "document_level": doc_entities,
            "per_chunk": entities_per_chunk
        },
        "importance": {
            "per_chunk": importance_per_chunk
        }
    }
    
    # Save back
    store.save(doc, chunks)
    return True
