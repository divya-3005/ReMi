import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = None

def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Set GROQ_API_KEY in .env")
        _client = Groq(api_key=api_key)
    return _client

def complete(system: str, user: str, max_tokens: int = 1024) -> str:
    """
    Combines system and user prompt and generates content using Groq's LLaMA 3.
    """
    client = get_client()
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_tokens=max_tokens,
        temperature=0.1
    )
    
    return response.choices[0].message.content
