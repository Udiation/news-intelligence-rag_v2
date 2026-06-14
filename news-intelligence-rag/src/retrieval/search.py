"""Phase 3b — the hybrid temporal searcher.

Glues the store (Phase 2) to the fusion math (``fusion.py``): pull candidates
from both retrievers, fuse with RRF, apply recency decay, and return the
top-N fused chunks ready for reranking.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from loguru import logger

from src.config import Settings, get_settings
from src.domain import Chunk, RetrievedChunk
from src.indexing.vector_store import HybridVectorStore
from src.retrieval.fusion import temporal_fusion


class HybridSearcher:
    """Runs sparse + dense retrieval and temporal fusion over a store."""

    def __init__(
        self,
        store: HybridVectorStore,
        settings: Optional[Settings] = None,
    ) -> None:
        self._store = store
        self._settings = settings or get_settings()

    def search(
        self,
        query: str,
        *,
        top_n: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> List[RetrievedChunk]:
        """Return the temporally-fused top-N chunks for ``query``.

        Gracefully returns ``[]`` when the store is empty or the query is blank.
        """
        if self._store.is_empty:
            logger.warning("Search called on an empty store; returning no results.")
            return []
        if not query or not query.strip():
            logger.debug("Blank query; returning no results.")
            return []

        k = self._settings.retrieval_top_k
        sparse_hits = self._store.search_sparse(query, k)
        dense_hits = self._store.search_dense(query, k)

        sparse_ranked: List[Chunk] = [chunk for chunk, _ in sparse_hits]
        dense_ranked: List[Chunk] = [chunk for chunk, _ in dense_hits]

        logger.debug(
            "Retrieved sparse={} dense={} candidates for query {!r}",
            len(sparse_ranked),
            len(dense_ranked),
            query,
        )

        fused = temporal_fusion(
            [sparse_ranked, dense_ranked],
            k=self._settings.rrf_k,
            half_life_hours=self._settings.time_decay_halflife_hours,
            now=now,
            missing_weight=self._settings.missing_timestamp_weight,
            top_n=top_n or self._settings.rerank_top_n,
        )
        logger.info("Fused to {} candidates for query {!r}", len(fused), query)
        return fused
