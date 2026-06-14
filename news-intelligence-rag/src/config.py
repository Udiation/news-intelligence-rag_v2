"""Central, 12-factor configuration for the News Intelligence RAG service.

All tunables are exposed as environment variables (optionally via a ``.env``
file) and parsed/validated through :class:`pydantic_settings.BaseSettings`.
Import :func:`get_settings` everywhere rather than re-reading the environment;
it is ``lru_cache``-d so the settings object is a process-wide singleton.

Example
-------
>>> from src.config import get_settings
>>> settings = get_settings()
>>> settings.rrf_k
60
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application configuration.

    Every field maps to an upper-cased environment variable of the same name
    (e.g. ``rrf_k`` -> ``RRF_K``). Defaults are production-sane.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    rss_feeds: List[str] = Field(
        default=[
            "http://feeds.bbci.co.uk/news/world/rss.xml",
            "https://feeds.reuters.com/reuters/worldNews",
            "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        ],
        description="RSS/Atom feed URLs to ingest.",
    )
    scrape_concurrency: int = Field(
        default=8, ge=1, le=64, description="Max simultaneous feed fetches."
    )
    scrape_timeout_seconds: float = Field(default=15.0, gt=0.0)
    scrape_max_retries: int = Field(default=3, ge=0, le=10)
    user_agent: str = Field(
        default="NewsIntelligenceRAG/1.0 (+https://github.com/Udiation)",
        description="HTTP User-Agent for polite scraping.",
    )

    # ------------------------------------------------------------------ #
    # Deduplication (MinHash-LSH)
    # ------------------------------------------------------------------ #
    dedup_jaccard_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    dedup_num_perm: int = Field(
        default=128, ge=16, description="MinHash permutations (accuracy vs. speed)."
    )
    dedup_shingle_size: int = Field(
        default=3, ge=1, description="Word n-gram size for shingling."
    )

    # ------------------------------------------------------------------ #
    # Chunking
    # ------------------------------------------------------------------ #
    chunk_size: int = Field(default=512, ge=64)
    chunk_overlap: int = Field(default=64, ge=0)

    # ------------------------------------------------------------------ #
    # Indexing / models
    # ------------------------------------------------------------------ #
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    nli_model: str = Field(default="microsoft/deberta-v3-small")
    hnsw_m: int = Field(default=32, ge=4, description="HNSW graph degree (M).")
    hnsw_ef_construction: int = Field(default=200, ge=8)
    hnsw_ef_search: int = Field(default=64, ge=1)
    index_dir: Path = Field(default=Path("artifacts/index"))

    # ------------------------------------------------------------------ #
    # Retrieval / fusion
    # ------------------------------------------------------------------ #
    retrieval_top_k: int = Field(
        default=50, ge=1, description="Candidates pulled per retriever before fusion."
    )
    rrf_k: int = Field(default=60, ge=1, description="RRF smoothing constant k.")
    time_decay_halflife_hours: float = Field(
        default=72.0, gt=0.0, description="Recency half-life H (hours)."
    )
    missing_timestamp_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Recency weight applied when published_at is unknown.",
    )
    rerank_top_n: int = Field(
        default=10, ge=1, description="Survivors after cross-encoder rerank."
    )

    # ------------------------------------------------------------------ #
    # NLI thresholds
    # ------------------------------------------------------------------ #
    nli_entailment_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    nli_contradiction_threshold: float = Field(default=0.50, ge=0.0, le=1.0)

    # ------------------------------------------------------------------ #
    # Serving / ops
    # ------------------------------------------------------------------ #
    log_level: str = Field(default="INFO")
    retrieval_only_mode: bool = Field(
        default=False,
        description="If true, skip loading rerank/NLI models and disable /answer.",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_below_size(cls, v: int, info) -> int:  # type: ignore[no-untyped-def]
        size = info.data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

    @field_validator("rss_feeds")
    @classmethod
    def _non_empty_feeds(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one RSS feed must be configured.")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance."""
    return Settings()
