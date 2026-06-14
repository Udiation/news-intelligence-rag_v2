"""Phase 1a — concurrent, fault-tolerant RSS/Atom scraping.

Design notes
------------
* **Concurrency** is bounded by an :class:`asyncio.Semaphore` so we never open
  more than ``scrape_concurrency`` sockets at once (politeness + resource caps).
* **Resilience** is provided by `tenacity`: transient network errors and 5xx
  responses are retried with exponential backoff. A feed that fails *all*
  retries is logged and skipped — one bad feed never aborts the crawl.
* **Timestamps** are parsed leniently; anything unparseable yields
  ``published_at=None`` (handled downstream by a neutral recency weight) rather
  than dropping the article.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp
import feedparser  # type: ignore[import-untyped]
from dateutil import parser as date_parser
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Settings, get_settings
from src.domain import Article


class FeedFetchError(RuntimeError):
    """Raised when a feed cannot be retrieved after all retries."""


def _parse_timestamp(raw: Optional[str]) -> Optional[datetime]:
    """Parse an RSS date string into a tz-aware UTC datetime, or ``None``."""
    if not raw:
        return None
    try:
        dt = date_parser.parse(raw)
    except (ValueError, OverflowError, TypeError):
        logger.debug("Unparseable timestamp: {!r}", raw)
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class RSSScraper:
    """Fetches and parses a list of RSS/Atom feeds concurrently."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._semaphore = asyncio.Semaphore(self._settings.scrape_concurrency)

    async def scrape(self) -> List[Article]:
        """Scrape every configured feed and return a flat list of articles."""
        timeout = aiohttp.ClientTimeout(total=self._settings.scrape_timeout_seconds)
        headers = {"User-Agent": self._settings.user_agent}

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            tasks = [
                self._scrape_one(session, url) for url in self._settings.rss_feeds
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        articles: List[Article] = []
        for url, result in zip(self._settings.rss_feeds, results):
            if isinstance(result, BaseException):
                logger.warning("Feed failed permanently, skipping {}: {}", url, result)
                continue
            articles.extend(result)

        logger.info(
            "Scraped {} articles from {} feeds",
            len(articles),
            len(self._settings.rss_feeds),
        )
        return articles

    async def _scrape_one(
        self, session: aiohttp.ClientSession, url: str
    ) -> List[Article]:
        async with self._semaphore:
            raw = await self._fetch_with_retry(session, url)
        return self._parse_feed(raw, source_url=url)

    @property
    def _retry_decorator(self):  # type: ignore[no-untyped-def]
        # Built from settings so max_attempts is configurable.
        return retry(
            reraise=True,
            stop=stop_after_attempt(self._settings.scrape_max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
            retry=retry_if_exception_type(
                (aiohttp.ClientError, asyncio.TimeoutError, FeedFetchError)
            ),
        )

    async def _fetch_with_retry(
        self, session: aiohttp.ClientSession, url: str
    ) -> bytes:
        async def _do_fetch() -> bytes:
            logger.debug("Fetching feed {}", url)
            async with session.get(url) as response:
                if response.status >= 500:
                    raise FeedFetchError(f"{url} returned {response.status}")
                if response.status >= 400:
                    # 4xx is not retryable — fail fast.
                    raise FeedFetchError(f"{url} returned {response.status} (no retry)")
                return await response.read()

        wrapped = self._retry_decorator(_do_fetch)
        return await wrapped()

    def _parse_feed(self, raw: bytes, source_url: str) -> List[Article]:
        parsed = feedparser.parse(raw)
        if parsed.bozo:  # malformed XML — feedparser still salvages entries
            logger.warning(
                "Malformed feed {} ({}); parsing best-effort entries",
                source_url,
                getattr(parsed, "bozo_exception", "unknown"),
            )

        source = parsed.feed.get("title", source_url) if parsed.feed else source_url
        articles: List[Article] = []
        for entry in parsed.entries:
            link = entry.get("link", "")
            title = entry.get("title", "").strip()
            if not link and not title:
                continue  # skip entries with no identity at all

            summary = entry.get("summary", "")
            content = summary
            if entry.get("content"):
                # Atom feeds often expose richer bodies here.
                content = entry["content"][0].get("value", summary)

            published = _parse_timestamp(
                entry.get("published") or entry.get("updated")
            )
            articles.append(
                Article(
                    title=title,
                    summary=summary,
                    content=content,
                    url=link,
                    source=source,
                    published_at=published,
                )
            )

        logger.debug("Parsed {} entries from {}", len(articles), source_url)
        return articles


async def scrape_all(settings: Optional[Settings] = None) -> List[Article]:
    """Convenience async entrypoint used by the indexing build step."""
    return await RSSScraper(settings).scrape()
