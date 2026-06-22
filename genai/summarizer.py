from dataclasses import dataclass
from typing import List, Dict, Any
from storage.document_store import DocumentStore
from genai.prompts import SUMMARIZE_CHUNK_SYSTEM, SUMMARIZE_DOC_SYSTEM
from genai.client import complete

@dataclass
class DocumentSummary:
    doc_id: str
    source_file: str
    chunk_summaries: List[str]
    full_summary: str
    top_entities: List[str]
    top_keywords: List[str]

def summarize_document(doc_id: str, store: DocumentStore) -> DocumentSummary:
    """
    Summarizes a full document using a hierarchical approach.
    """
    res = store.get_by_id(doc_id)
    if not res:
        raise ValueError(f"Document {doc_id} not found.")
        
    doc, chunks = res
    source_file = doc.filename
    
    if not chunks:
        # Graceful handling for 0 chunks
        return DocumentSummary(
            doc_id=doc_id,
            source_file=source_file,
            chunk_summaries=[],
            full_summary="Document has no text chunks to summarize.",
            top_entities=[],
            top_keywords=[]
        )

    chunk_summaries = []
    for chunk in chunks:
        summary = complete(system=SUMMARIZE_CHUNK_SYSTEM, user=chunk.content, max_tokens=150)
        chunk_summaries.append(summary)
        
    current_summaries = chunk_summaries.copy()
    while len(current_summaries) > 20:
        next_level = []
        for i in range(0, len(current_summaries), 20):
            batch = current_summaries[i:i+20]
            batch_text = "\n\n".join([f"Summary {j+1}: {s}" for j, s in enumerate(batch)])
            batch_summary = complete(
                system="You are an expert summarizer. Combine the following summaries into a single comprehensive summary block.", 
                user=batch_text, 
                max_tokens=300
            )
            next_level.append(batch_summary)
        current_summaries = next_level

    nlp_data = doc.nlp or {}
    
    top_keywords = []
    if "keywords" in nlp_data and "document_level" in nlp_data["keywords"]:
        top_keywords = [kw["keyword"] for kw in nlp_data["keywords"]["document_level"][:10]]
        
    top_entities = []
    if "entities" in nlp_data and "document_level" in nlp_data["entities"]:
        top_entities = [f"{ent['text']} ({ent['label']})" for ent in nlp_data["entities"]["document_level"][:10]]

    doc_prompt_parts = []
    doc_prompt_parts.append("CHUNK SUMMARIES:")
    for i, s in enumerate(current_summaries):
        doc_prompt_parts.append(f"[{i+1}] {s}")
        
    if top_keywords:
        doc_prompt_parts.append("\nTOP KEYWORDS:")
        doc_prompt_parts.append(", ".join(top_keywords))
        
    if top_entities:
        doc_prompt_parts.append("\nTOP ENTITIES:")
        doc_prompt_parts.append(", ".join(top_entities))
        
    final_user_prompt = "\n".join(doc_prompt_parts)
    
    full_summary = complete(system=SUMMARIZE_DOC_SYSTEM, user=final_user_prompt, max_tokens=1024)
    
    return DocumentSummary(
        doc_id=doc_id,
        source_file=source_file,
        chunk_summaries=chunk_summaries,
        full_summary=full_summary,
        top_entities=top_entities,
        top_keywords=top_keywords
    )
