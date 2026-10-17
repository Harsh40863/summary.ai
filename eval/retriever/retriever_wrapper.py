"""LangChain-style retriever wrapper around backend DocumentSearchEngine with Cross-Encoder reranking."""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch

from langchain_core.documents import Document

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_BACKEND_DIR = os.path.join(_PROJECT_ROOT, "backend")
_SUMMARY_AI_DIR = os.path.join(_PROJECT_ROOT, "summary.ai")
for _path in (_BACKEND_DIR, _SUMMARY_AI_DIR, _PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import nltk
_local_nltk = os.path.join(_PROJECT_ROOT, ".nltk_data")
if os.path.isdir(_local_nltk) and _local_nltk not in nltk.data.path:
    nltk.data.path.insert(0, _local_nltk)

from sentence_transformers import CrossEncoder

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

CROSS_ENCODER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def _format_eval_documents(input_documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    return [
        {"name": doc["doc_name"], "content": doc["content"]}
        for doc in input_documents
    ]


class AppDocumentRetriever:
    """
    Two-stage Reranking Retriever for evaluation:
    1. Over-retrieves candidates using fast vector search (fetch_k).
    2. Reranks candidate chunks with Cross-Encoder (cross-attention) to eliminate noise.
    3. Retains top_k highest scoring chunks.
    """

    _cached_reranker: Optional[CrossEncoder] = None

    def __init__(
        self,
        input_documents: List[Dict[str, str]],
        *,
        threshold: float = 0.35,
        top_k: int = 2,
        fetch_k: int = 10,
        enable_reranking: bool = True,
    ):
        self.threshold = threshold
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.enable_reranking = enable_reranking and (
            os.getenv("ENABLE_RERANKING", "true").lower() in ("true", "1", "yes")
        )
        self._engine = self._build_engine(input_documents)
        if self.enable_reranking:
            if AppDocumentRetriever._cached_reranker is None:
                AppDocumentRetriever._cached_reranker = CrossEncoder(CROSS_ENCODER_MODEL)
            self._reranker = AppDocumentRetriever._cached_reranker
        else:
            self._reranker = None

    def _build_engine(self, input_documents: List[Dict[str, str]]):
        from backend.logic import DocumentSearchEngine

        db_docs = _format_eval_documents(input_documents)

        def _noop_rag(_query: str, _chunk: str, _name: str) -> str:
            return ""

        with patch(
            "backend.logic.fetch_documents_from_db", return_value=db_docs
        ), patch("backend.logic.generate_answer_with_rag", side_effect=_noop_rag):
            return DocumentSearchEngine(user_email="__eval__")

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self, query: str) -> List[Document]:
        # 1. OVER-RETRIEVE: broader candidate pool with relaxed threshold
        results = self._engine.find_relevant_documents(
            query, threshold=0.10 if self.enable_reranking else self.threshold
        )
        candidates: List[Document] = []
        fetch_limit = self.fetch_k if self.enable_reranking else self.top_k
        for item in results[:fetch_limit]:
            chunk = item.get("relevant_chunk") or ""
            if not chunk.strip():
                continue
            doc_meta = item.get("doc") or {}
            candidates.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_name": doc_meta.get("name"),
                        "score": item.get("score"),
                        "doc_index": item.get("doc_index"),
                    },
                )
            )

        if not candidates:
            return []

        # 2. RERANK: score each (query, chunk) pair together using Cross-Encoder
        if self.enable_reranking and self._reranker is not None and len(candidates) > 1:
            pairs = [(query, doc.page_content) for doc in candidates]
            scores = self._reranker.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            surviving = []
            for doc, score in ranked[: self.top_k]:
                doc.metadata["rerank_score"] = float(score)
                surviving.append(doc)
            return surviving

        return candidates[: self.top_k]


def retriever_for_golden(golden: Dict[str, Any]) -> AppDocumentRetriever:
    """Factory used by eval runner and live RagPipeline."""
    return AppDocumentRetriever(
        golden["input_documents"],
        threshold=golden.get("expected_threshold", 0.35),
        top_k=int(os.getenv("RERANKER_TOP_K", str(golden.get("top_k", 2)))),
        fetch_k=int(os.getenv("RERANKER_FETCH_K", "10")),
        enable_reranking=True,
    )


RetrieverBuilder = Callable[[Dict[str, Any]], AppDocumentRetriever]
