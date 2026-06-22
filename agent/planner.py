import uuid
import re
from typing import List
from models.research import SubQuestion
from genai.client import complete

PLANNER_SYSTEM = """You are an expert research planner.
Your task is to decompose a broad research question into 3 to 5 specific, non-overlapping sub-questions.
These sub-questions should guide an evidence-retrieval system to gather all necessary facts to answer the main question.

Format your response as a numbered list. Example:
1. What are the short-term cognitive effects?
2. What are the cardiovascular effects?
3. What is the recommended daily intake?
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
