"""Phase 2a — text chunking.

Wraps LangChain's :class:`RecursiveCharacterTextSplitter` to split article
bodies into overlapping chunks while **carrying every article's metadata onto
every chunk**. Downstream retrieval returns chunks, but the API surfaces the
source/url/timestamp so answers stay attributable.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.config import Settings, get_settings
from src.domain import Article, Chunk


class Chunker:
    """Splits articles into metadata-rich :class:`Chunk` objects."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_article(self, article: Article) -> List[Chunk]:
        """Chunk a single article. Returns ``[]`` for empty bodies."""
        text = article.text
        if not text.strip():
            logger.debug("Skipping empty article: {}", article.url or article.title)
            return []

        pieces = self._splitter.split_text(text)
        return [
            Chunk.from_article(article, piece, position)
            for position, piece in enumerate(pieces)
            if piece.strip()
        ]

    def chunk_articles(self, articles: List[Article]) -> List[Chunk]:
        """Chunk a corpus of articles into a flat list of chunks."""
        chunks: List[Chunk] = []
        for article in articles:
            chunks.extend(self.chunk_article(article))
        logger.info("Chunked {} articles into {} chunks", len(articles), len(chunks))
        return chunks
