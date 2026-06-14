"""Phase 5a — public API contract (pydantic v2).

These models are the *only* validated boundary between callers and the
pipeline. Internal dataclasses (:mod:`src.domain`) are converted to/from these
at the edge so the wire format stays stable independently of internals.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class SearchRequest(BaseModel):
    """Retrieval-only request."""

    query: str = Field(..., min_length=1, max_length=2_000, examples=["chip export curbs"])
    top_k: int = Field(default=5, ge=1, le=50)


class AnswerRequest(SearchRequest):
    """Full-pipeline request (retrieval + grounded answer + contradictions)."""

    include_contradictions: bool = Field(default=True)


# --------------------------------------------------------------------------- #
# Shared response components
# --------------------------------------------------------------------------- #
class RetrievedChunkModel(BaseModel):
    chunk_id: str
    text: str
    title: str
    url: str
    source: str
    published_at: Optional[datetime] = None
    score: float
    retriever: str


class CitationModel(BaseModel):
    hypothesis_sentence: str
    chunk_id: str
    source: str
    url: str
    entailment_score: float = Field(..., ge=0.0, le=1.0)


class ContradictionModel(BaseModel):
    chunk_id: str
    source: str
    url: str
    premise_snippet: str
    hypothesis_sentence: str
    contradiction_score: float = Field(..., ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class SearchResponse(BaseModel):
    query: str
    results: List[RetrievedChunkModel]
    warning: Optional[str] = Field(
        default=None,
        description="Set (with HTTP 200) when results are degraded, e.g. empty index.",
    )


class AnswerResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationModel]
    contradictions: List[ContradictionModel]
    results: List[RetrievedChunkModel]
    warning: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok", "degraded"])
    indices_loaded: bool
    index_size: int
    retrieval_only_mode: bool
