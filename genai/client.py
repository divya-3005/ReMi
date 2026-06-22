import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def configure_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in .env — free at aistudio.google.com")
    genai.configure(api_key=api_key)

def complete(system: str, user: str, max_tokens: int = 1024) -> str:
    """
    Combines system and user prompt and generates content using gemini-1.5-flash.
    """
    configure_client()
    
    model = genai.GenerativeModel("gemini-flash-latest")
    full_prompt = f"{system}\n\n{user}"
    
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens
    )
    
    response = model.generate_content(full_prompt, generation_config=generation_config)
    return response.text
