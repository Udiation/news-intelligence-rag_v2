"""Phase 5b — the FastAPI serving application.

Lifecycle
---------
On startup (lifespan) we load the persisted hybrid index and — unless
``retrieval_only_mode`` is set — the rerank/NLI models, holding them in
``app.state``. If the index directory is missing the service still boots; it
reports ``degraded`` health and every query returns an empty result set with a
``warning`` (HTTP 200) instead of a 500. This keeps a misconfigured deploy
debuggable rather than crash-looping.

Endpoints
---------
* ``GET  /health``  liveness + index/model status
* ``POST /search``  hybrid temporal retrieval only
* ``POST /answer``  full pipeline (disabled in retrieval-only mode -> 503)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from src.config import Settings, get_settings
from src.domain import RetrievedChunk
from src.generation.validator import ValidationPipeline
from src.indexing.vector_store import HybridVectorStore
from src.logging_config import configure_logging
from src.retrieval.search import HybridSearcher
from src.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    CitationModel,
    ContradictionModel,
    HealthResponse,
    RetrievedChunkModel,
    SearchRequest,
    SearchResponse,
)

_EMPTY_INDEX_WARNING = (
    "Index is empty or not loaded. Build it with "
    "`python -m src.indexing.vector_store --build`."
)


def _to_chunk_models(chunks: List[RetrievedChunk]) -> List[RetrievedChunkModel]:
    return [
        RetrievedChunkModel(
            chunk_id=rc.chunk.chunk_id,
            text=rc.chunk.text,
            title=rc.chunk.title,
            url=rc.chunk.url,
            source=rc.chunk.source,
            published_at=rc.chunk.published_at,
            score=rc.score,
            retriever=rc.retriever,
        )
        for rc in chunks
    ]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting News Intelligence RAG API")

    store = HybridVectorStore(settings).load()
    app.state.settings = settings
    app.state.store = store
    app.state.searcher = HybridSearcher(store, settings)

    app.state.validation: Optional[ValidationPipeline] = None
    if settings.retrieval_only_mode:
        logger.warning("retrieval_only_mode=True; /answer disabled, NLI/rerank skipped.")
    else:
        try:
            app.state.validation = ValidationPipeline(settings=settings)
            logger.info("Validation pipeline ready (rerank + NLI lazy-loaded).")
        except Exception as exc:  # noqa: BLE001 - never let model setup kill the app
            logger.error("Validation pipeline init failed; serving retrieval-only: {}", exc)
            app.state.validation = None

    if store.is_empty:
        logger.warning(_EMPTY_INDEX_WARNING)

    yield
    logger.info("Shutting down News Intelligence RAG API")


app = FastAPI(
    title="News Intelligence RAG",
    description="Hybrid temporal retrieval with contradiction detection.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on {} {}: {}", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    store: HybridVectorStore = request.app.state.store
    settings: Settings = request.app.state.settings
    loaded = not store.is_empty
    return HealthResponse(
        status="ok" if loaded else "degraded",
        indices_loaded=loaded,
        index_size=store.size,
        retrieval_only_mode=settings.retrieval_only_mode,
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: Request, payload: SearchRequest) -> SearchResponse:
    searcher: HybridSearcher = request.app.state.searcher
    results = searcher.search(payload.query, top_n=payload.top_k)
    warning = _EMPTY_INDEX_WARNING if request.app.state.store.is_empty else None
    return SearchResponse(
        query=payload.query,
        results=_to_chunk_models(results),
        warning=warning,
    )


@app.post("/answer", response_model=AnswerResponse)
async def answer(request: Request, payload: AnswerRequest) -> AnswerResponse:
    validation: Optional[ValidationPipeline] = request.app.state.validation
    if validation is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Answer generation is unavailable (retrieval-only mode).",
        )

    searcher: HybridSearcher = request.app.state.searcher
    candidates = searcher.search(payload.query, top_n=payload.top_k)

    if not candidates:
        return AnswerResponse(
            query=payload.query,
            answer="",
            citations=[],
            contradictions=[],
            results=[],
            warning=_EMPTY_INDEX_WARNING
            if request.app.state.store.is_empty
            else "No relevant documents found for the query.",
        )

    result = validation.run(payload.query, candidates)
    contradictions = (
        [
            ContradictionModel(
                chunk_id=c.chunk_id,
                source=c.source,
                url=c.url,
                premise_snippet=c.premise_snippet,
                hypothesis_sentence=c.hypothesis_sentence,
                contradiction_score=c.contradiction_score,
            )
            for c in result.contradictions
        ]
        if payload.include_contradictions
        else []
    )

    return AnswerResponse(
        query=payload.query,
        answer=result.answer,
        citations=[
            CitationModel(
                hypothesis_sentence=cit.hypothesis_sentence,
                chunk_id=cit.chunk_id,
                source=cit.source,
                url=cit.url,
                entailment_score=cit.entailment_score,
            )
            for cit in result.citations
        ],
        contradictions=contradictions,
        results=_to_chunk_models(result.reranked),
    )
