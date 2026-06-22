from models.research import Finding, LinkedAnswer, LinkedSentence
from grounding.extractor import extract_spans, split_into_sentences
from typing import List

def link(answer: str, findings: List[Finding]) -> LinkedAnswer:
    # Consolidate all unique sources from findings
    unique_sources = {}
    for finding in findings:
        for src in finding.sources:
            key = f"{src.source_file}_{src.chunk_index}"
            unique_sources[key] = src
            
    chunks = list(unique_sources.values())
    
    # Extract best span for each sentence
    spans = extract_spans(answer, chunks)
    answer_sentences = split_into_sentences(answer)
    
    linked_sentences = []
    
    # If chunks was empty, spans might be empty
    if not spans:
        for sentence in answer_sentences:
            linked_sentences.append(LinkedSentence(sentence=sentence, evidence=[], grounded=False))
        return LinkedAnswer(answer=answer, sentences=linked_sentences)
        
    for i, sentence in enumerate(answer_sentences):
        span = spans[i]
        # Check relevance
        is_grounded = span.relevance_score > 0.4
        linked_sentences.append(
            LinkedSentence(
                sentence=sentence,
                evidence=[span] if span.relevance_score > 0 else [],
                grounded=is_grounded
            )
        )
        
    return LinkedAnswer(answer=answer, sentences=linked_sentences)
