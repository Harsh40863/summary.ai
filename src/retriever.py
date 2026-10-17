"""
src/retriever.py
================
Two-stage Reranking Retriever for summary.ai with native LangSmith tracing.
1. Over-retrieves a broad candidate pool using fast bi-encoder vector search (fetch_k).
2. Reranks candidate chunks with Cross-Encoder (cross-attention) to eliminate noise.
3. Retains top_k highest scoring chunks for the generator.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"
_SUMMARY_AI_DIR = _PROJECT_ROOT / "summary.ai"

for _p in (_BACKEND_DIR, _SUMMARY_AI_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_BACKEND_DIR / ".env")

import nltk
_local_nltk = _PROJECT_ROOT / ".nltk_data"
if _local_nltk.is_dir() and str(_local_nltk) not in nltk.data.path:
    nltk.data.path.insert(0, str(_local_nltk))

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

# Small, fast, CPU-friendly reranker (~80MB, cached locally)
CROSS_ENCODER = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def load_store(user_email: Optional[str] = None, input_documents: Optional[List[Dict[str, Any]]] = None):
    """
    Builds a FAISS vectorstore over document chunks using LocalEmbeddingAdapter.
    """
    from backend.logic import DocumentSearchEngine, LocalEmbeddingAdapter
    from unittest.mock import patch
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        from langchain.vectorstores import FAISS

    if input_documents:
        def _noop_rag(_query: str, _chunk: str, _name: str) -> str:
            return ""

        with patch("backend.logic.fetch_documents_from_db", return_value=input_documents), \
             patch("backend.logic.generate_answer_with_rag", side_effect=_noop_rag):
            engine = DocumentSearchEngine(user_email="__eval__")
    else:
        engine = DocumentSearchEngine(user_email=user_email)

    chunks = engine._build_document_chunks()
    if not chunks:
        # Return an empty store placeholder
        return FAISS.from_texts([""], LocalEmbeddingAdapter())

    docs = [
        Document(
            page_content=chunk["text"],
            metadata={
                "doc_index": chunk["doc_index"],
                "chunk_index": chunk["chunk_index"],
                "best_sentence": chunk.get("best_sentence", ""),
                "doc_name": engine.documents[chunk["doc_index"]].get("name", "document.pdf")
                if chunk["doc_index"] < len(engine.documents)
                else "document.pdf",
            },
        )
        for chunk in chunks
    ]
    return FAISS.from_documents(docs, LocalEmbeddingAdapter())


class RerankingRetriever:
    """
    Two-Stage Cross-Encoder Reranking Retriever with LangSmith Tracing.
    """

    def __init__(
        self,
        store: Any = None,
        fetch_k: int = 10,
        top_k: int = 3,
        model_name: str = CROSS_ENCODER,
    ):
        self.store = store
        self.reranker = CrossEncoder(model_name)
        self.fetch_k = fetch_k  # How many the bi-encoder brings back (over-retrieve)
        self.top_k = top_k      # How many survive after cross-encoder reranking

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self, query: str) -> List[Document]:
        # 1. OVER-RETRIEVE: fast bi-encoder, deliberately more than we need
        if self.store is None:
            self.store = load_store()

        if hasattr(self.store, "similarity_search"):
            candidates = self.store.similarity_search(query, k=self.fetch_k)
        elif hasattr(self.store, "invoke"):
            candidates = self.store.invoke(query)
        else:
            candidates = []

        if not candidates:
            return []

        # 2. RERANK: score each (query, chunk) pair TOGETHER with the cross-encoder
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # 3. SORT by score, keep the best top_k
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        surviving = []
        for doc, score in ranked[: self.top_k]:
            if hasattr(doc, "metadata"):
                doc.metadata["rerank_score"] = float(score)
            surviving.append(doc)

        return surviving
