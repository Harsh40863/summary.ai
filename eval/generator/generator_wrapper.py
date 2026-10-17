"""
Generator wrapper for summary.ai RAG pipeline.
Coordinates retrieval and answer synthesis for golden test cases.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"
_SUMMARY_AI_DIR = _PROJECT_ROOT / "summary.ai"

for _p in (_BACKEND_DIR, _SUMMARY_AI_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import nltk
_local_nltk = _PROJECT_ROOT / ".nltk_data"
if _local_nltk.is_dir() and str(_local_nltk) not in nltk.data.path:
    nltk.data.path.insert(0, str(_local_nltk))

from eval.retriever.retriever_wrapper import AppDocumentRetriever, retriever_for_golden


class AppRAGGenerator:
    """
    RAG Generator wrapper for evaluation:
    1. Uses AppDocumentRetriever to retrieve relevant chunks.
    2. Synthesizes an answer using the app's generate_answer_with_rag.
    """

    def __init__(self, golden: Dict[str, Any]):
        self.golden = golden
        self.retriever = retriever_for_golden(golden)

    def invoke(self, query: str) -> Tuple[str, List[str]]:
        """
        Executes end-to-end RAG pipeline for the given query.
        Returns:
            Tuple of (generated_answer, retrieved_chunk_texts)
        """
        # 1. Retrieve chunks
        docs = self.retriever.invoke(query)
        chunks = [d.page_content for d in docs if d.page_content.strip()]
        doc_name = (
            docs[0].metadata.get("doc_name", "document.pdf")
            if docs
            else "document.pdf"
        )

        # 2. Check for empty retrieval or out-of-domain query
        if not chunks:
            fallback = (
                "Based on the provided documents, there is no relevant information available to answer this question."
            )
            return fallback, []

        context = "\n\n".join(chunks)

        # 3. Call generator with optimized prompt
        from src.generator import generate

        answer = generate(query, chunks, doc_name)
        return answer, chunks


def generator_for_golden(golden: Dict[str, Any]) -> AppRAGGenerator:
    """Factory to create an AppRAGGenerator instance for a golden test case."""
    return AppRAGGenerator(golden)
