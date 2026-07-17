"""
src/config.py
─────────────
Single source of truth for all configuration.

Design decision: model names live ONLY here, never in client.py or anywhere
downstream. Google's deprecation cadence (1.5-flash → 404, 2.0-flash retired
June 2026, 2.5-flash scheduled October 2026) made model-name abstraction a
first-class requirement, not a nice-to-have. Swapping models is a one-line
change here — no code changes, no test changes required elsewhere.

All fields can be overridden via environment variables (same name, uppercase).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ── API credentials ──────────────────────────────────────────────────────
    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "")
    )
    groq_api_key: str = field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY", "")
    )

    # ── Model names (ONLY place in the codebase that references these) ────────
    # Check https://ai.google.dev/gemini-api/docs/models before each deploy —
    # Google retires models on a 3–6 month cadence.
    gemini_llm_model: str = field(
        default_factory=lambda: os.environ.get("GEMINI_LLM_MODEL", "gemini-2.5-flash")
    )
    gemini_embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        )
    )
    groq_llm_model: str = field(
        default_factory=lambda: os.environ.get(
            "GROQ_LLM_MODEL", "llama-3.1-8b-instant"
        )
    )

    # ── RAG hyperparameters ───────────────────────────────────────────────────
    chunk_size: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_SIZE", "800"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", "150"))
    )
    retrieval_k: int = field(
        default_factory=lambda: int(os.environ.get("RETRIEVAL_K", "10"))
    )
    rrf_k_constant: int = field(
        default_factory=lambda: int(os.environ.get("RRF_K_CONSTANT", "60"))
    )
    embedding_dim: int = field(
        default_factory=lambda: int(os.environ.get("EMBEDDING_DIM", "768"))
    )  # gemini-embedding-001 = 768. Update if you switch embedding models.

    # ── Agent thresholds ──────────────────────────────────────────────────────
    # analyzer_relevance_floor is FIXED across all workflow attempts.
    # Reformulating sub-questions is the retry lever, not loosening this filter.
    analyzer_relevance_floor: float = field(
        default_factory=lambda: float(
            os.environ.get("ANALYZER_RELEVANCE_FLOOR", "0.40")
        )
    )
    min_citation_coverage: float = field(
        default_factory=lambda: float(
            os.environ.get("MIN_CITATION_COVERAGE", "0.55")
        )
    )
    min_citation_utilization: float = field(
        default_factory=lambda: float(
            os.environ.get("MIN_CITATION_UTILIZATION", "0.30")
        )
    )
    # Set to 0 on demo day to disable retry and conserve free-tier API credits.
    max_workflow_retries: int = field(
        default_factory=lambda: int(os.environ.get("MAX_WORKFLOW_RETRIES", "1"))
    )

    # ── Request handling ──────────────────────────────────────────────────────
    research_timeout_seconds: int = field(
        default_factory=lambda: int(
            os.environ.get("RESEARCH_TIMEOUT_SECONDS", "120")
        )
    )

    def validate(self) -> None:
        """Raise ValueError for any critically missing configuration."""
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Uses lru_cache so the dataclass is constructed exactly once per process.
    In tests, call get_settings.cache_clear() before patching environment
    variables to force a fresh read.
    """
    settings = Settings()
    return settings
