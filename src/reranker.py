"""
src/reranker.py
===============
Two-stage Cross-Encoder Reranker for summary.ai RAG pipeline.
Takes candidate documents from initial vector retrieval and reranks them
using joint cross-attention for deep semantic precision.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


class CrossEncoderReranker:
    """
    Cross-Encoder Reranker using sentence-transformers CrossEncoder.
    Provides deep semantic cross-attention between (query, document) pairs.
    """

    _instance: Optional["CrossEncoderReranker"] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        default_top_k: int = 2,
        min_score_threshold: Optional[float] = None,
    ):
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.default_top_k = int(os.getenv("RERANKER_TOP_K", str(default_top_k)))
        _env_thresh = os.getenv("RERANKER_SCORE_THRESHOLD")
        if _env_thresh and _env_thresh.strip().lower() not in ("none", "false", ""):
            self.min_score_threshold = float(_env_thresh)
        else:
            self.min_score_threshold = min_score_threshold
        self._model = None

    @classmethod
    def get_instance(cls) -> "CrossEncoderReranker":
        if cls._instance is None:
            cls._instance = CrossEncoderReranker()
        return cls._instance

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def is_enabled(self) -> bool:
        return os.getenv("ENABLE_RERANKING", "true").lower() in ("true", "1", "yes")

    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None,
    ) -> List[Any]:
        """
        Reranks a list of candidate documents (LangChain Document, dict, or str).

        Args:
            query: The user query string.
            documents: List of candidate documents.
            top_k: Number of top documents to return.

        Returns:
            Reranked and sorted documents with rerank scores.
        """
        if not documents:
            return []

        k = top_k or self.default_top_k
        if len(documents) <= 1 and self.min_score_threshold is None:
            return documents[:k]

        # Extract text representations
        doc_texts = []
        for doc in documents:
            if hasattr(doc, "page_content"):
                doc_texts.append(doc.page_content)
            elif isinstance(doc, dict):
                doc_texts.append(doc.get("text") or doc.get("content") or doc.get("page_content", ""))
            else:
                doc_texts.append(str(doc))

        try:
            model = self._load_model()
            pairs = [[query, text] for text in doc_texts]
            scores = model.predict(pairs)

            # Attach scores to documents
            scored_docs = []
            for doc, score, text in zip(documents, scores, doc_texts):
                score_float = float(score)
                # Filter out obvious distractors if score is well below threshold
                if self.min_score_threshold is not None and score_float < self.min_score_threshold:
                    continue

                if hasattr(doc, "metadata"):
                    doc.metadata["rerank_score"] = score_float
                elif isinstance(doc, dict):
                    doc["rerank_score"] = score_float

                scored_docs.append((doc, score_float))

            # Sort descending by rerank score
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            reranked = [d for d, _ in scored_docs[:k]]

            # Fallback if all were below threshold
            if not reranked and documents:
                reranked = [documents[0]]

            return reranked

        except Exception as e:
            # Fallback gracefully to original order on model failure
            print(f"Warning: Reranker failed ({e}), falling back to vector ranking.")
            return documents[:k]


@traceable(name="reranker_step", run_type="parser")
def rerank_documents(
    query: str,
    documents: List[Any],
    top_k: Optional[int] = None,
) -> List[Any]:
    """LangSmith traceable reranker function."""
    reranker = CrossEncoderReranker.get_instance()
    if not reranker.is_enabled():
        k = top_k or reranker.default_top_k
        return documents[:k]
    return reranker.rerank(query, documents, top_k=top_k)
