"""
src/rag_pipeline.py
===================
End-to-end RAG pipeline for summary.ai with native LangSmith tracing.
Combines document search (retrieval) + answer generation with full trace linkage.
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

# Synchronize LangSmith / LangChain tracing environment variables
for _k_src, _k_dst in (("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"), ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")):
    if os.getenv(_k_src) and not os.getenv(_k_dst):
        os.environ[_k_dst] = os.getenv(_k_src).strip('"\'')
if os.getenv("LANGSMITH_TRACING") == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT").strip('"\'')

import nltk
_local_nltk = _PROJECT_ROOT / ".nltk_data"
if _local_nltk.is_dir() and str(_local_nltk) not in nltk.data.path:
    nltk.data.path.insert(0, str(_local_nltk))

from eval.retriever.retriever_wrapper import AppDocumentRetriever, retriever_for_golden
from src.generator import generate
from src.reranker import rerank_documents

# LangSmith tracing integration (safe fallback if langsmith is not installed)
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


@traceable(name="retriever_step", run_type="retriever")
def _retrieve_step(target_case: Dict[str, Any], query: str) -> List[Any]:
    """Retrieves relevant chunks with LangSmith retriever tracing."""
    retriever = retriever_for_golden(target_case)
    return retriever.invoke(query)


@traceable(name="generator_step", run_type="llm")
def _generate_step(query: str, context: List[str], doc_name: str) -> str:
    """Synthesizes answer with LangSmith LLM tracing."""
    return generate(query, context, doc_name=doc_name)


class RagPipeline:
    """
    End-to-end RAG Pipeline with hierarchical LangSmith tracing:
    1. Retrieve relevant chunks using DocumentSearchEngine / AppDocumentRetriever
    2. Synthesize grounded answer using generator
    3. Expose trace linkage (query -> retrieved chunks -> answer)
    """

    def __init__(
        self,
        golden: Optional[Dict[str, Any]] = None,
        threshold: float = 0.35,
        top_k: int = 3,
    ):
        self.golden = golden
        self.threshold = threshold
        self.top_k = top_k

    @traceable(name="summary_ai_rag_pipeline", run_type="chain")
    def invoke(
        self,
        query: str,
        golden: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG pipeline:
        Retrieve -> Rerank/Filter -> Generate
        """
        target_case = golden or self.golden

        if target_case is not None:
            # Stage 1: Vector Retrieval
            raw_docs = _retrieve_step(target_case, query)
            # Stage 2: Cross-Encoder Reranking
            docs = rerank_documents(query, raw_docs, top_k=self.top_k)

            context = [d.page_content for d in docs if d.page_content.strip()]
            doc_name = (
                docs[0].metadata.get("doc_name", "document.pdf")
                if docs
                else "document.pdf"
            )
            doc_metas = [
                {
                    "content": d.page_content,
                    "score": d.metadata.get("score"),
                    "rerank_score": d.metadata.get("rerank_score"),
                    "doc_name": d.metadata.get("doc_name"),
                }
                for d in docs
            ]
        else:
            docs = []
            context = []
            doc_name = "document.pdf"
            doc_metas = []

        # Generate answer using our high-precision/recall generator
        answer = _generate_step(query, context, doc_name)

        try:
            from langsmith.run_helpers import get_current_run_tree
            _rt = get_current_run_tree()
            run_id = str(_rt.id) if _rt else None
        except Exception:
            run_id = None

        return {
            "query": query,
            "answer": answer,
            "context": context,
            "doc_name": doc_name,
            "retrieved_docs": doc_metas,
            "run_id": run_id,
        }

