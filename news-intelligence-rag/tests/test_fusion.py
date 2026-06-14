"""Unit tests for the pure ranking math (RRF + time decay)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from src.domain import Chunk
from src.retrieval.fusion import (
    reciprocal_rank_fusion,
    temporal_fusion,
    time_decay_weight,
)


def _chunk(cid: str, published_at=None) -> Chunk:
    return Chunk(
        chunk_id=cid,
        article_id=cid,
        text=f"text {cid}",
        title=cid,
        url=f"http://x/{cid}",
        source="test",
        published_at=published_at,
        position=0,
    )


def test_rrf_rewards_consensus_top_ranks() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    # 'a' is rank-1 in both lists -> should score highest.
    scores = reciprocal_rank_fusion([[a, b, c], [a, c, b]], k=60)
    assert scores["a"] == max(scores.values())
    assert math.isclose(scores["a"], 2.0 / 61.0)


def test_time_decay_halves_at_half_life() -> None:
    now = datetime(2025, 1, 4, tzinfo=timezone.utc)
    published = now - timedelta(hours=72)
    w = time_decay_weight(published, half_life_hours=72.0, now=now)
    assert math.isclose(w, 0.5, rel_tol=1e-9)


def test_time_decay_missing_timestamp_is_neutral() -> None:
    assert time_decay_weight(None, half_life_hours=72.0, missing_weight=1.0) == 1.0


def test_future_timestamp_clamped_to_one() -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    future = now + timedelta(hours=5)
    assert time_decay_weight(future, half_life_hours=72.0, now=now) == 1.0


def test_temporal_fusion_prefers_fresh_on_tie() -> None:
    now = datetime(2025, 1, 10, tzinfo=timezone.utc)
    fresh = _chunk("fresh", published_at=now)
    stale = _chunk("stale", published_at=now - timedelta(days=30))
    # Identical rank positions across lists -> recency breaks the tie.
    fused = temporal_fusion([[fresh, stale], [fresh, stale]], k=60,
                            half_life_hours=72.0, now=now)
    assert fused[0].chunk.chunk_id == "fresh"


def test_temporal_fusion_empty_input() -> None:
    assert temporal_fusion([[], []]) == []
