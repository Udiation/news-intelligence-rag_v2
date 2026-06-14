"""Phase 3a — Reciprocal Rank Fusion and exponential recency decay.

This module is intentionally pure (no I/O, no models) so the ranking math is
unit-testable in isolation. It mirrors the formulae in the README:

    RRF(d)        = sum_r 1 / (k + rank_r(d))
    w_time(d)     = exp(-lambda * age_hours),   lambda = ln(2) / H
    score(d)      = RRF(d) * w_time(d)
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from src.domain import Chunk, RetrievedChunk

_LN2 = math.log(2.0)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Chunk]],
    k: int = 60,
) -> Dict[str, float]:
    """Fuse several ranked lists into a single ``chunk_id -> RRF score`` map.

    Parameters
    ----------
    ranked_lists:
        Each inner sequence is one retriever's results, **ordered best-first**.
    k:
        RRF smoothing constant. Larger ``k`` flattens the reward for top ranks.

    Returns
    -------
    Mapping from ``chunk_id`` to its summed reciprocal-rank score. A chunk
    absent from a list simply contributes nothing from that list.
    """
    if k < 1:
        raise ValueError("RRF constant k must be >= 1")

    scores: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):  # 1-based ranks
            scores[chunk.chunk_id] += 1.0 / (k + rank)
    return dict(scores)


def time_decay_weight(
    published_at: Optional[datetime],
    *,
    half_life_hours: float,
    now: Optional[datetime] = None,
    missing_weight: float = 1.0,
) -> float:
    """Exponential recency weight in ``(0, 1]`` for an article timestamp.

    ``missing_weight`` is returned when ``published_at`` is ``None`` (unknown
    age) so undated articles are neither boosted nor unfairly buried. Future
    timestamps (clock skew) are clamped to age 0 -> weight 1.0.
    """
    if published_at is None:
        return missing_weight
    if half_life_hours <= 0.0:
        raise ValueError("half_life_hours must be > 0")

    reference = now or datetime.now(timezone.utc)
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    age_hours = (reference - published_at).total_seconds() / 3600.0
    if age_hours <= 0.0:
        return 1.0

    lam = _LN2 / half_life_hours
    return math.exp(-lam * age_hours)


def temporal_fusion(
    ranked_lists: Sequence[Sequence[Chunk]],
    *,
    k: int = 60,
    half_life_hours: float = 72.0,
    now: Optional[datetime] = None,
    missing_weight: float = 1.0,
    top_n: Optional[int] = None,
) -> List[RetrievedChunk]:
    """Run RRF, apply recency decay, and return chunks sorted by final score.

    Returns an empty list if every input list is empty.
    """
    # Deduplicate chunk objects by id while preserving the richest instance.
    by_id: Dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for chunk in ranked:
            by_id.setdefault(chunk.chunk_id, chunk)

    if not by_id:
        return []

    rrf_scores = reciprocal_rank_fusion(ranked_lists, k=k)

    fused: List[Tuple[Chunk, float]] = []
    for chunk_id, rrf in rrf_scores.items():
        chunk = by_id[chunk_id]
        weight = time_decay_weight(
            chunk.published_at,
            half_life_hours=half_life_hours,
            now=now,
            missing_weight=missing_weight,
        )
        fused.append((chunk, rrf * weight))

    # Deterministic ordering: score desc, then chunk_id for stable tie-breaks.
    fused.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))

    if top_n is not None:
        fused = fused[:top_n]

    return [RetrievedChunk(chunk=c, score=s, retriever="fused") for c, s in fused]
