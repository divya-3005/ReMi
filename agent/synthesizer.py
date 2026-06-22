from typing import List
from models.research import Finding
from genai.client import complete

SYNTHESIZER_SYSTEM = """You are an expert research analyst.
You have been provided with a research question and several findings based on evidence retrieved from a document corpus.

Your task is to synthesize a structured final report using ONLY the findings provided.
If the findings contradict each other, note the contradiction. Do NOT hallucinate external information.
Whenever you state a fact, you MUST cite the source file inline, for example: [source: file1.pdf].

Format your response exactly with these markdown sections:

# Executive Summary
(A 2-3 sentence high-level answer to the research question based on the findings)

# Detailed Findings
(Synthesize the findings systematically. Use sub-headings for each major theme or sub-question answered)

# Confidence Assessment
(Provide an overall assessment of the evidence quality: High, Medium, or Low. Explain why in 1 sentence based on the amount and relevance of evidence)
"""

def synthesize(question: str, findings: List[Finding]) -> str:
    """
    Synthesizes a structured final report from the gathered findings.
    """
    if not findings:
        return "No sufficient evidence was found to answer the research question."
        
    # Build prompt context
    context_parts = []
    context_parts.append(f"RESEARCH QUESTION: {question}\n")
    
    all_sources = set()
    
    for i, finding in enumerate(findings):
        context_parts.append(f"--- FINDING {i+1} ---")
        context_parts.append(f"Answer: {finding.answer}")
        
        # Track all unique sources
        for src in finding.sources:
            all_sources.add(src.source_file)
            
    user_prompt = "\n".join(context_parts)
    
    report = complete(system=SYNTHESIZER_SYSTEM, user=user_prompt, max_tokens=1500)
    
    # Append the Evidence Sources section manually
    report += "\n\n# Evidence Sources\n"
    if all_sources:
        for src in sorted(list(all_sources)):
            report += f"- {src}\n"
    else:
        report += "No specific sources cited.\n"
        
    return report
