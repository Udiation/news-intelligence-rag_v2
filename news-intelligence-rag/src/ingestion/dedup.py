"""Phase 1b — near-duplicate removal via MinHash + LSH.

Wire services (AP, Reuters, AFP) are republished verbatim across many outlets.
Indexing all copies wastes space and lets a single story dominate retrieval.
We estimate pairwise Jaccard similarity of word-shingle sets with MinHash and
use an LSH index to find candidates in ~O(n) instead of O(n^2), keeping only
the first occurrence of each near-duplicate cluster (Jaccard >= threshold).
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

from datasketch import MinHash, MinHashLSH  # type: ignore[import-untyped]
from loguru import logger

from src.config import Settings, get_settings
from src.domain import Article

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _shingles(text: str, size: int) -> Set[str]:
    """Return the set of word n-gram shingles for ``text``."""
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < size:
        # Too short to shingle — treat the whole token list as one shingle.
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def _minhash(shingles: Set[str], num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for shingle in shingles:
        mh.update(shingle.encode("utf-8"))
    return mh


class MinHashDeduplicator:
    """Drops near-duplicate articles using a MinHash-LSH index."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    def deduplicate(self, articles: List[Article]) -> List[Article]:
        """Return articles with near-duplicates removed (order preserved).

        The first article in each duplicate cluster is kept; later copies are
        dropped. Articles with empty text are passed through untouched (they
        cannot be meaningfully compared and are handled downstream).
        """
        if not articles:
            logger.info("Dedup received 0 articles; nothing to do.")
            return []

        threshold = self._settings.dedup_jaccard_threshold
        num_perm = self._settings.dedup_num_perm
        shingle_size = self._settings.dedup_shingle_size

        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        kept: List[Article] = []
        dropped = 0

        for idx, article in enumerate(articles):
            shingles = _shingles(article.text, shingle_size)
            if not shingles:
                kept.append(article)
                continue

            mh = _minhash(shingles, num_perm)
            if lsh.query(mh):  # at least one near-duplicate already indexed
                dropped += 1
                logger.debug("Dropping near-duplicate: {}", article.url or article.title)
                continue

            key = f"art-{idx}"
            lsh.insert(key, mh)
            kept.append(article)

        logger.info(
            "Dedup: kept {} / {} articles (dropped {} duplicates, threshold={})",
            len(kept),
            len(articles),
            dropped,
            threshold,
        )
        return kept
