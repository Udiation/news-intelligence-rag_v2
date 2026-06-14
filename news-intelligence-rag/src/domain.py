"""Shared domain models passed between pipeline stages.

These are plain, immutable-ish dataclasses (not pydantic) because they live on
the hot internal path and never cross the network boundary. The pydantic models
in :mod:`src.api.schemas` are the public, validated contract; these are the
internal representation.

Flow of types through the pipeline::

    Article  --chunk-->  Chunk  --retrieve-->  RetrievedChunk
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


def _stable_id(*parts: str) -> str:
    """Deterministic short id from arbitrary string parts (SHA-1 prefix)."""
    digest = hashlib.sha1("\u241f".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass(slots=True)
class Article:
    """A single news article as scraped from an RSS/Atom feed.

    Attributes
    ----------
    article_id:
        Deterministic id derived from the canonical URL (stable across runs).
    title, summary, content:
        Textual fields. ``content`` may equal ``summary`` for feeds that do not
        expose full bodies.
    url, source:
        Canonical link and the feed/source label.
    published_at:
        Timezone-aware publication time, or ``None`` if unparseable/missing.
    """

    title: str
    summary: str
    content: str
    url: str
    source: str
    published_at: Optional[datetime]
    article_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.article_id:
            self.article_id = _stable_id(self.url or self.title)

    @property
    def text(self) -> str:
        """Concatenated title + body used for indexing and dedup."""
        body = self.content or self.summary
        return f"{self.title}\n\n{body}".strip()


@dataclass(slots=True)
class Chunk:
    """A retrievable text chunk carrying its article's metadata."""

    chunk_id: str
    article_id: str
    text: str
    title: str
    url: str
    source: str
    published_at: Optional[datetime]
    position: int  # 0-based chunk index within the article

    @classmethod
    def from_article(cls, article: "Article", text: str, position: int) -> "Chunk":
        return cls(
            chunk_id=_stable_id(article.article_id, str(position)),
            article_id=article.article_id,
            text=text,
            title=article.title,
            url=article.url,
            source=article.source,
            published_at=article.published_at,
            position=position,
        )


@dataclass(slots=True)
class RetrievedChunk:
    """A :class:`Chunk` paired with a relevance score and provenance flag."""

    chunk: Chunk
    score: float
    retriever: str = "fused"  # "bm25" | "dense" | "fused" | "rerank"
