"""Phase 2b — the hybrid (sparse + dense) vector store.

Builds and persists **two** indices over the same chunk corpus:

* a **sparse** BM25 (Okapi) index via ``rank_bm25``;
* a **dense** FAISS **HNSW** graph over ``all-MiniLM-L6-v2`` embeddings, using
  inner-product on L2-normalized vectors (i.e. cosine similarity).

The store is the artifact boundary between offline and online: ``build`` +
``save`` run in batch; ``load`` + ``search_*`` run on the request path.

Empty-index safety is a first-class concern: ``build`` refuses to serialize an
empty corpus, and all search methods return ``[]`` (never raise) when the
store is empty.

Run as a script to (re)build from live feeds::

    python -m src.indexing.vector_store --build
"""

from __future__ import annotations

import argparse
import asyncio
import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple

import faiss  # type: ignore[import-untyped]
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
from sentence_transformers import SentenceTransformer

from src.config import Settings, get_settings
from src.domain import Chunk
from src.logging_config import configure_logging

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_FAISS_FILE = "dense.faiss"
_BM25_FILE = "sparse.pkl"
_CHUNKS_FILE = "chunks.pkl"


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer shared by BM25 indexing and querying."""
    return _TOKEN_RE.findall(text.lower())


class EmptyIndexError(RuntimeError):
    """Raised when attempting to build/save an index from zero chunks."""


class HybridVectorStore:
    """Owns the sparse + dense indices and the backing chunk corpus.

    Construct, then either :meth:`build` from chunks or :meth:`load` from disk.
    The embedding model is loaded lazily on first use so importing this module
    is cheap.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._embedder: Optional[SentenceTransformer] = None
        self._chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._faiss: Optional[faiss.Index] = None

    # ----------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------- #
    @property
    def is_empty(self) -> bool:
        """True if there is nothing to search."""
        return len(self._chunks) == 0 or self._faiss is None or self._bm25 is None

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            logger.info("Loading embedding model {}", self._settings.embedding_model)
            self._embedder = SentenceTransformer(self._settings.embedding_model)
        return self._embedder

    # ----------------------------------------------------------------- #
    # Build
    # ----------------------------------------------------------------- #
    def build(self, chunks: List[Chunk]) -> None:
        """Build both indices in memory from ``chunks``.

        Raises
        ------
        EmptyIndexError
            If ``chunks`` is empty — we never persist a useless index.
        """
        if not chunks:
            raise EmptyIndexError("Cannot build an index from zero chunks.")

        self._chunks = chunks
        logger.info("Building hybrid index over {} chunks", len(chunks))

        # --- Sparse (BM25) ---
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

        # --- Dense (FAISS HNSW) ---
        embeddings = self._encode([c.text for c in chunks])
        dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, self._settings.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self._settings.hnsw_ef_construction
        index.hnsw.efSearch = self._settings.hnsw_ef_search
        index.add(embeddings)
        self._faiss = index

        logger.info("Hybrid index built (dim={}, vectors={})", dim, index.ntotal)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into L2-normalized float32 embeddings for cosine/IP."""
        vectors = self.embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit vectors -> IP == cosine
        ).astype(np.float32)
        return np.ascontiguousarray(vectors)

    # ----------------------------------------------------------------- #
    # Persistence
    # ----------------------------------------------------------------- #
    def save(self, directory: Optional[Path] = None) -> None:
        """Serialize the store to ``directory`` (defaults to settings)."""
        if self.is_empty or self._faiss is None or self._bm25 is None:
            raise EmptyIndexError("Refusing to save an empty index.")

        directory = directory or self._settings.index_dir
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._faiss, str(directory / _FAISS_FILE))
        with (directory / _BM25_FILE).open("wb") as fh:
            pickle.dump(self._bm25, fh)
        with (directory / _CHUNKS_FILE).open("wb") as fh:
            pickle.dump(self._chunks, fh)

        logger.info("Saved index to {} ({} chunks)", directory, len(self._chunks))

    def load(self, directory: Optional[Path] = None) -> "HybridVectorStore":
        """Load a previously-saved store. Leaves the store empty on miss.

        Returns ``self`` for chaining. A missing index directory is *not* an
        error — the service can boot with an empty store and report degraded
        health rather than crash-loop.
        """
        directory = directory or self._settings.index_dir
        faiss_path = directory / _FAISS_FILE
        bm25_path = directory / _BM25_FILE
        chunks_path = directory / _CHUNKS_FILE

        if not (faiss_path.exists() and bm25_path.exists() and chunks_path.exists()):
            logger.warning(
                "Index artifacts missing under {}; starting with an EMPTY store.",
                directory,
            )
            self._chunks, self._bm25, self._faiss = [], None, None
            return self

        try:
            self._faiss = faiss.read_index(str(faiss_path))
            self._faiss.hnsw.efSearch = self._settings.hnsw_ef_search
            with bm25_path.open("rb") as fh:
                self._bm25 = pickle.load(fh)
            with chunks_path.open("rb") as fh:
                self._chunks = pickle.load(fh)
        except (OSError, pickle.UnpicklingError) as exc:
            logger.error("Failed to load index from {}: {}", directory, exc)
            self._chunks, self._bm25, self._faiss = [], None, None
            return self

        logger.info("Loaded index from {} ({} chunks)", directory, len(self._chunks))
        return self

    # ----------------------------------------------------------------- #
    # Search
    # ----------------------------------------------------------------- #
    def search_sparse(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        """Top-``k`` BM25 hits as ``(chunk, score)``; ``[]`` if empty."""
        if self.is_empty or self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            (self._chunks[i], float(scores[i]))
            for i in top_idx
            if scores[i] > 0.0
        ]

    def search_dense(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        """Top-``k`` dense (cosine) hits as ``(chunk, score)``; ``[]`` if empty."""
        if self.is_empty or self._faiss is None:
            return []
        q = self._encode([query])
        k_eff = min(k, self._faiss.ntotal)
        scores, indices = self._faiss.search(q, k_eff)
        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS pads with -1 when fewer than k neighbours exist
                continue
            results.append((self._chunks[int(idx)], float(score)))
        return results


def _build_from_feeds(settings: Settings) -> None:
    """Offline pipeline: scrape -> dedup -> chunk -> index -> persist."""
    # Imported here to keep the online import graph light.
    from src.indexing.chunker import Chunker
    from src.ingestion.dedup import MinHashDeduplicator
    from src.ingestion.scraper import scrape_all

    articles = asyncio.run(scrape_all(settings))
    if not articles:
        logger.error("No articles scraped; aborting build.")
        return

    deduped = MinHashDeduplicator(settings).deduplicate(articles)
    chunks = Chunker(settings).chunk_articles(deduped)
    if not chunks:
        logger.error("No chunks produced after chunking; aborting build.")
        return

    store = HybridVectorStore(settings)
    store.build(chunks)
    store.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="News Intelligence RAG index builder")
    parser.add_argument(
        "--build", action="store_true", help="Scrape feeds and (re)build the index."
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    if args.build:
        _build_from_feeds(settings)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
