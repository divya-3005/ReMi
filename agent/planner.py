import uuid
import re
from typing import List
from models.research import SubQuestion
from genai.client import complete

PLANNER_SYSTEM = """You are an expert research planner for an AI extraction system.
Your task is to decompose a user's research question into specific, literal sub-questions to search a vector database.

CRITICAL RULES:
1. If the user's question is a simple, direct factoid (e.g., "Who handled X?", "What is Y?"), do NOT decompose it. Just return the original question as the single sub-question.
2. NEVER hallucinate constraints or roles that aren't in the original question. Do NOT generate questions about the "primary owner", "creator", or "maintenance team".
3. Ask direct, simple fact-finding questions.
4. If the user asks a broad cross-document question (e.g. "across all projects"), focus your sub-questions on extracting the raw data points required, rather than generalizing into abstract theory.

Format your response as a numbered list. Example for a complex question:
1. What are the names associated with X?
2. What is the value of metric Y?

Example for a simple factoid:
1. Who handled the Tableau Dashboard for the Financial Fraud project?
"""

def plan(research_question: str) -> List[SubQuestion]:
    """
    Decomposes a research question into sub-questions.
    """
    user_prompt = f"Research Question: {research_question}\n\nPlease decompose this into 3 to 5 sub-questions."
    
    response_text = complete(system=PLANNER_SYSTEM, user=user_prompt, max_tokens=300)
    
    # Parse numbered list
    sub_questions = []
    for line in response_text.split('\n'):
        line = line.strip()
        # Match lines like "1. Question?" or "1) Question?"
        match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
        if match:
            question_text = match.group(1).strip()
            # Remove any markdown bolding
            question_text = question_text.replace('**', '')
            if question_text:
                sq = SubQuestion(
                    id=str(uuid.uuid4()),
                    question=question_text,
                    status="pending"
                )
                sub_questions.append(sq)
                
    # Fallback if parsing fails
    if not sub_questions:
        sub_questions.append(
            SubQuestion(id=str(uuid.uuid4()), question=research_question, status="pending")
        )
        
    return sub_questions
