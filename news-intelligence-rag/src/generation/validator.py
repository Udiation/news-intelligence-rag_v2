"""Phase 4 — validation, reranking, citation grounding, contradiction detection.

Three concerns live here, each independently optional so partial model
outages degrade gracefully:

1. :class:`Reranker` — re-scores fused candidates with a cross-encoder
   (full query-document attention) for precision at the top.
2. :class:`AnswerGenerator` (protocol) + :class:`ExtractiveAnswerGenerator` —
   produces the *hypothesis* sentence(s). The default is a dependency-free
   extractive baseline; swap in an LLM-backed implementation in production by
   satisfying the protocol.
3. :class:`NLIValidator` — runs ``deberta-v3`` NLI between each retrieved
   premise and the generated hypothesis to (a) ground citations via entailment
   and (b) surface cross-source contradictions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

import numpy as np
from loguru import logger

from src.config import Settings, get_settings
from src.domain import RetrievedChunk

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Citation:
    """A hypothesis sentence grounded in a supporting premise chunk."""

    hypothesis_sentence: str
    chunk_id: str
    source: str
    url: str
    entailment_score: float


@dataclass(slots=True)
class Contradiction:
    """A premise chunk that contradicts the generated answer."""

    chunk_id: str
    source: str
    url: str
    premise_snippet: str
    hypothesis_sentence: str
    contradiction_score: float


@dataclass(slots=True)
class ValidationResult:
    """Output of the full Phase-4 validation pass."""

    answer: str
    citations: List[Citation]
    contradictions: List[Contradiction]
    reranked: List[RetrievedChunk]


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #
class Reranker:
    """Cross-encoder reranker. Lazily loads the model on first use."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._model = None  # type: ignore[var-annotated]

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker {}", self._settings.reranker_model)
            self._model = CrossEncoder(self._settings.reranker_model)

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """Return candidates re-sorted by cross-encoder relevance to ``query``."""
        if not candidates:
            return []
        self._ensure_model()
        assert self._model is not None

        pairs = [[query, c.chunk.text] for c in candidates]
        scores = self._model.predict(pairs)  # higher == more relevant
        order = np.argsort(scores)[::-1]

        reranked = [
            RetrievedChunk(
                chunk=candidates[i].chunk,
                score=float(scores[i]),
                retriever="rerank",
            )
            for i in order
        ]
        top_n = self._settings.rerank_top_n
        logger.debug("Reranked {} -> top {}", len(candidates), min(top_n, len(reranked)))
        return reranked[:top_n]


# --------------------------------------------------------------------------- #
# Answer generation (hypothesis)
# --------------------------------------------------------------------------- #
class AnswerGenerator(Protocol):
    """Pluggable contract for producing the answer (NLI hypothesis).

    Replace the default extractive generator with an LLM-backed one by
    implementing this single method.
    """

    def generate(self, query: str, context: Sequence[RetrievedChunk]) -> str:
        ...


class ExtractiveAnswerGenerator:
    """Dependency-free baseline: stitch the top reranked chunks into an answer.

    This is deliberately honest — there is no hidden LLM. It returns the most
    relevant chunk text(s), which the NLI stage then grounds and contradiction-
    checks. Production swaps in an LLM via the :class:`AnswerGenerator` protocol.
    """

    def __init__(self, max_chunks: int = 2) -> None:
        self._max_chunks = max_chunks

    def generate(self, query: str, context: Sequence[RetrievedChunk]) -> str:
        if not context:
            return ""
        selected = context[: self._max_chunks]
        return " ".join(c.chunk.text.strip() for c in selected).strip()


# --------------------------------------------------------------------------- #
# NLI contradiction / entailment
# --------------------------------------------------------------------------- #
def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


class NLIValidator:
    """DeBERTa-v3 NLI for citation grounding and contradiction detection.

    Label order for MNLI-style heads is assumed ``[contradiction, neutral,
    entailment]`` and verified against ``model.config.id2label`` at load time,
    remapping if the checkpoint differs.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._tokenizer = None  # type: ignore[var-annotated]
        self._model = None  # type: ignore[var-annotated]
        self._idx_contradiction = 0
        self._idx_entailment = 2

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401  (imported for side-effect availability)
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        name = self._settings.nli_model
        logger.info("Loading NLI model {}", name)
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForSequenceClassification.from_pretrained(name)
        self._model.eval()
        self._resolve_label_indices()

    def _resolve_label_indices(self) -> None:
        """Map class indices from the checkpoint's id2label, robust to ordering."""
        id2label = getattr(self._model.config, "id2label", None)
        if not id2label:
            return
        for idx, label in id2label.items():
            low = str(label).lower()
            if "contradict" in low:
                self._idx_contradiction = int(idx)
            elif "entail" in low:
                self._idx_entailment = int(idx)
        logger.debug(
            "NLI label map: contradiction={} entailment={}",
            self._idx_contradiction,
            self._idx_entailment,
        )

    def _probabilities(self, premises: List[str], hypotheses: List[str]) -> np.ndarray:
        """Return softmax probabilities of shape ``(N, num_labels)``."""
        import torch

        assert self._tokenizer is not None and self._model is not None
        encoded = self._tokenizer(
            premises,
            hypotheses,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**encoded).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def validate(
        self,
        answer: str,
        premises: Sequence[RetrievedChunk],
    ) -> tuple[List[Citation], List[Contradiction]]:
        """Ground ``answer`` against ``premises`` and flag contradictions."""
        citations: List[Citation] = []
        contradictions: List[Contradiction] = []

        hypotheses = _split_sentences(answer)
        if not hypotheses or not premises:
            return citations, contradictions

        self._ensure_model()

        ent_thr = self._settings.nli_entailment_threshold
        con_thr = self._settings.nli_contradiction_threshold

        # Score the cross product of (premise chunk, hypothesis sentence).
        for premise in premises:
            prem_text = premise.chunk.text
            probs = self._probabilities([prem_text] * len(hypotheses), hypotheses)
            for hyp, row in zip(hypotheses, probs):
                p_ent = float(row[self._idx_entailment])
                p_con = float(row[self._idx_contradiction])

                if p_ent >= ent_thr:
                    citations.append(
                        Citation(
                            hypothesis_sentence=hyp,
                            chunk_id=premise.chunk.chunk_id,
                            source=premise.chunk.source,
                            url=premise.chunk.url,
                            entailment_score=p_ent,
                        )
                    )
                if p_con >= con_thr:
                    contradictions.append(
                        Contradiction(
                            chunk_id=premise.chunk.chunk_id,
                            source=premise.chunk.source,
                            url=premise.chunk.url,
                            premise_snippet=prem_text[:240],
                            hypothesis_sentence=hyp,
                            contradiction_score=p_con,
                        )
                    )

        logger.info(
            "NLI validation: {} citations, {} contradictions",
            len(citations),
            len(contradictions),
        )
        return citations, contradictions


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class ValidationPipeline:
    """Rerank -> generate -> validate, wired together with graceful fallbacks."""

    def __init__(
        self,
        reranker: Optional[Reranker] = None,
        generator: Optional[AnswerGenerator] = None,
        nli: Optional[NLIValidator] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._reranker = reranker or Reranker(self._settings)
        self._generator = generator or ExtractiveAnswerGenerator()
        self._nli = nli or NLIValidator(self._settings)

    def run(self, query: str, candidates: Sequence[RetrievedChunk]) -> ValidationResult:
        if not candidates:
            return ValidationResult(answer="", citations=[], contradictions=[], reranked=[])

        reranked = self._reranker.rerank(query, candidates)
        answer = self._generator.generate(query, reranked)
        citations, contradictions = self._nli.validate(answer, reranked)
        return ValidationResult(
            answer=answer,
            citations=citations,
            contradictions=contradictions,
            reranked=reranked,
        )
