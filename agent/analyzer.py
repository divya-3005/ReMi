from typing import List
from models.research import Finding

def analyze_findings(findings: List[Finding], min_confidence: float = 0.3) -> List[Finding]:
    """
    Filters out low-confidence findings and ranks them.
    Deduplicates sources appearing across multiple findings.
    """
    cleaned_findings = []
    
    for finding in findings:
        if finding.confidence_score >= min_confidence:
            # Deduplicate sources within the finding itself
            seen_chunks = set()
            dedup_sources = []
            for src in finding.sources:
                key = f"{src.source_file}_{src.chunk_index}"
                if key not in seen_chunks:
                    seen_chunks.add(key)
                    dedup_sources.append(src)
            finding.sources = dedup_sources
            cleaned_findings.append(finding)
            
    # Rank findings by confidence score descending
    cleaned_findings.sort(key=lambda x: x.confidence_score, reverse=True)
    
    # Fallback: if all findings were rejected due to strict confidence thresholds 
    # (e.g., for generic 'overview' questions), retain the top finding anyway so 
    # the synthesizer has some baseline context to work with.
    if not cleaned_findings and findings:
        findings.sort(key=lambda x: x.confidence_score, reverse=True)
        top_finding = findings[0]
        
        seen_chunks = set()
        dedup_sources = []
        for src in top_finding.sources:
            key = f"{src.source_file}_{src.chunk_index}"
            if key not in seen_chunks:
                seen_chunks.add(key)
                dedup_sources.append(src)
                
        top_finding.sources = dedup_sources
        cleaned_findings.append(top_finding)
    
    return cleaned_findings
