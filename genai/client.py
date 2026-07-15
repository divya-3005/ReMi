import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = None

class TokenBucketRateLimiter:
    def __init__(self, rpm: int):
        self.capacity = rpm
        self.tokens = rpm
        self.rate = rpm / 60.0  # tokens per second
        self.last_update = time.time()

    def wait_if_needed(self):
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

        if self.tokens < 1.0:
            sleep_time = (1.0 - self.tokens) / self.rate
            time.sleep(sleep_time)
            self.tokens = 0.0
            self.last_update = time.time()
        else:
            self.tokens -= 1.0

# Conservatively default to 5 requests per minute
_rate_limiter = TokenBucketRateLimiter(rpm=5)

def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Set GROQ_API_KEY in .env")
        _client = Groq(api_key=api_key)
    return _client

class LLMRateLimitError(Exception): pass

def complete(system: str, user: str, max_tokens: int = 1024) -> str:
    """
    Combines system and user prompt and generates content using Groq's LLaMA 3.
    Applies client-side rate limiting to avoid hitting 429 quota limits, and
    exponential backoff if limits are still hit.
    """
    client = get_client()
    
    delays = [1, 2, 4, 8]
    for attempt, delay in enumerate(delays + [None]):
        _rate_limiter.wait_if_needed()
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str:
                if delay is not None:
                    time.sleep(delay)
                    continue
                else:
                    raise LLMRateLimitError(f"Groq API rate limit exceeded after {len(delays)} retries: {str(e)}")
            raise
