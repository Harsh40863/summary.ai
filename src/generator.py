"""
src/generator.py
================
Standard generator interface for summary.ai RAG application.
Exposes generate(query, context) -> answer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Union

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"
_SUMMARY_AI_DIR = _PROJECT_ROOT / "summary.ai"

for _p in (_BACKEND_DIR, _SUMMARY_AI_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_BACKEND_DIR / ".env")


def generate(
    query: str,
    context: Union[List[str], str],
    doc_name: str = "document.pdf",
) -> str:
    """
    Generate a grounded response using the supplied context and query.

    Args:
        query: The user prompt or question.
        context: Known-good context chunks (list of strings or text).
        doc_name: Document identifier.

    Returns:
        The generated answer string.
    """
    if isinstance(context, list):
        context_str = "\n\n".join(c for c in context if isinstance(c, str) and c.strip())
    else:
        context_str = str(context).strip()

    if not context_str:
        return "Based on the provided documents, there is no relevant information available to answer this question."

    # 1. Try application's native RAG generation logic
    try:
        from rag.rag_engine import generate_answer_with_rag

        answer = generate_answer_with_rag(query, context_str, doc_name)
        if (
            answer
            and not answer.strip().startswith("Error generating response:")
            and not answer.strip().startswith("**Context from")
        ):
            return answer
    except Exception:
        pass

    # 2. Try Groq generation if GROQ_API_KEY is available
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from openai import OpenAI

            groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
            client = OpenAI(
                api_key=groq_key,
                base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            )
            system_prompt = (
                "You are an expert factual assistant in a RAG system.\n"
                "Your objective is maximum Precision (zero hallucinations) and maximum Recall (complete coverage).\n\n"
                "RULES:\n"
                "1. STRICT GROUNDING: Rely ONLY on facts, numbers, and statements directly present in the context. "
                "Never invent, assume, or extrapolate.\n"
                "2. OUT-OF-DOMAIN HANDLING: If the context does NOT contain information to answer the question, or if the question is unrelated, "
                "respond EXACTLY: 'Based on the provided documents, there is no relevant information available to answer this question.'\n"
                "3. FULL COVERAGE: Answer all parts of the user's question thoroughly. Include every relevant figure, metric, percentage, entity, and detail.\n"
                "4. PARTIAL COVERAGE: If the context answers only part of the question, state the answered part completely and explicitly note that the remaining details are not available in the documents."
            )
            user_prompt = f"Context:\n{context_str}\n\nQuestion:\n{query}\n\nAnswer:"

            resp = client.chat.completions.create(
                model=groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            ans = resp.choices[0].message.content or ""
            if ans.strip():
                return ans.strip()
        except Exception:
            pass

    # 3. Clean fallback context extraction
    return f"According to '{doc_name}': {context_str}"
