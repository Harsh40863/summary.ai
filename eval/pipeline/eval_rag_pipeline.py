"""
eval_rag_pipeline.py
====================
End-to-end evaluation of the RAG pipeline using the RAG Triad.

THE RAG TRIAD & EVALUATION RUBRIC:
----------------------------------
1. CONTEXTUAL RELEVANCY (Retriever Signal-to-Noise):
   - Measures whether the retrieved context is relevant to the query.
   - Rubric:
     * 1.0 (Excellent): Context directly answers the query with minimal irrelevant noise.
     * 0.7-0.9 (Good): Context contains the answer, with small irrelevant background text.
     * 0.4-0.6 (Partial): Context touches the topic but misses key required facts.
     * 0.0-0.3 (Poor): Context is irrelevant or off-topic.

2. FAITHFULNESS (Generator Grounding / Zero Hallucination):
   - Measures whether the generated answer is strictly supported by the retrieved context.
   - Rubric:
     * 1.0 (Faithful): 100% of claims are factual and verified by context.
     * 0.7-0.9 (Minor Extrapolation): Claims are supported with slight stylistic variance.
     * 0.4-0.6 (Partially Hallucinated): Several claims lack context support.
     * 0.0-0.3 (Hallucinated): Answers contradict context or fabricate information.

3. ANSWER RELEVANCY (Generator Query-Alignment):
   - Measures whether the answer directly addresses the user query without fluff.
   - Rubric:
     * 1.0 (Direct & Clear): Answers the specific question directly.
     * 0.7-0.9 (Relevant): Answers the question but includes minor tangents.
     * 0.4-0.6 (Vague / Incomplete): Indirect answer or misses core intent.
     * 0.0-0.3 (Irrelevant): Off-topic answer.

Usage:
    python eval_rag_pipeline.py
    python -m eval.eval_rag_pipeline --category financial_report
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Ensure project root & eval dir are in sys.path
_PIPELINE_DIR = Path(__file__).resolve().parent
_EVAL_DIR = _PIPELINE_DIR.parent
_PROJECT_ROOT = _EVAL_DIR.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"
_SRC_DIR = _PROJECT_ROOT / "src"

for _p in (_PROJECT_ROOT, _BACKEND_DIR, _SRC_DIR, _EVAL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_BACKEND_DIR / ".env")

os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "Y"

import nltk
_local_nltk = _PROJECT_ROOT / ".nltk_data"
if _local_nltk.is_dir() and str(_local_nltk) not in nltk.data.path:
    nltk.data.path.insert(0, str(_local_nltk))

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from eval.harness import (
    load_goldens,
    print_summary,
    save_results_json,
    summarize_by_metric,
)
from src.rag_pipeline import RagPipeline

GOLDEN_PATH = _PROJECT_ROOT / "eval" / "datasets" / "retriever_golden_dataset.json"
DEFAULT_JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "groq")
THRESHOLD = float(os.getenv("DEEPEVAL_PIPELINE_THRESHOLD", "0.7"))


# ============================================================
# LLM JUDGE (Groq / Grok / OpenAI)
# ============================================================
class GroqJudgeModel(DeepEvalBaseLLM):
    """DeepEval judge model wrapper for GroqCloud fast inference."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        super().__init__(model_name=self.model_name)

    _semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self):
        import asyncio
        if GroqJudgeModel._semaphore is None:
            GroqJudgeModel._semaphore = asyncio.Semaphore(2)
        return GroqJudgeModel._semaphore

    def load_model(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=3)

    def load_async_model(self):
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=3)

    def supports_json_mode(self) -> bool:
        return True

    def generate(self, prompt: str, schema=None, **kwargs) -> str:
        client = self.load_model()
        import time
        params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if schema is not None or "json" in prompt.lower():
            params["response_format"] = {"type": "json_object"}

        for attempt in range(8):
            try:
                response = client.chat.completions.create(**params)
                time.sleep(0.5)
                return response.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "429" in err_str:
                    time.sleep(3.0 * (attempt + 1))
                elif attempt < 2:
                    time.sleep(1.0)
                else:
                    raise e
        return ""

    def generate_with_schema(self, prompt: str, schema=None, **kwargs) -> str:
        return self.generate(prompt, schema=schema, **kwargs)

    async def a_generate(self, prompt: str) -> str:
        import asyncio
        sem = self._get_semaphore()
        async with sem:
            client = self.load_async_model()
            for attempt in range(8):
                try:
                    response = await client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    await asyncio.sleep(0.4)
                    return response.choices[0].message.content or ""
                except Exception as e:
                    err_str = str(e).lower()
                    if "rate_limit" in err_str or "429" in err_str:
                        await asyncio.sleep(3.0 * (attempt + 1))
                    elif attempt < 2:
                        await asyncio.sleep(1.0)
                    else:
                        raise e
            return ""

    def get_model_name(self) -> str:
        return self.model_name



def _resolve_judge(judge_name: str):
    if "groq" in judge_name.lower() or (
        os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY")
    ):
        model = None if "groq" in judge_name.lower() else judge_name
        return GroqJudgeModel(model_name=model)
    return judge_name


# ============================================================
# RUN EVALUATION ON END-TO-END PIPELINE
# ============================================================
def run(
    rag: RagPipeline,
    golden_path: Path = GOLDEN_PATH,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    threshold: float = THRESHOLD,
    category: Optional[str] = None,
    skip_negative: bool = True,
) -> dict:
    # 1. LOAD queries
    goldens = load_goldens(golden_path)

    if category:
        goldens = [g for g in goldens if g.get("category") == category]
    if skip_negative:
        goldens = [g for g in goldens if g.get("category") != "negative_test"]

    judge = _resolve_judge(judge_model)
    judge_label = getattr(judge, "model_name", str(judge_model))

    print(f"\n======================================================================")
    print(f"  RAG PIPELINE END-TO-END EVALUATION (RAG TRIAD)")
    print(f"======================================================================")
    print(f"Evaluating {len(goldens)} cases on live pipeline...")
    print(f"Judge: {judge_label} | Pass Threshold: {threshold:.2f}\n")

    # 2. RUN THE INJECTED PIPELINE per query, build test case from LIVE output
    test_cases = []
    for idx, g in enumerate(goldens, 1):
        # Execute live pipeline: Retrieve -> Generate
        result = rag.invoke(g["query"], golden=g)
        answer = result["answer"]
        context = result["context"]

        print(f"  [{idx:02d}/{len(goldens):02d}] {g['id']} ({g.get('category')}): retrieved {len(context)} chunks -> generated answer")

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,
                retrieval_context=context if context else ["No relevant context found."],
                expected_output=g.get("ideal_answer"),
            )
        )

    # 3. THE THREE TRIAD METRICS
    metrics = [
        ContextualRelevancyMetric(
            threshold=threshold,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
        FaithfulnessMetric(
            threshold=threshold,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
        AnswerRelevancyMetric(
            threshold=threshold,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
    ]

    hyperparameters = {
        "evaluation_type": "rag_triad_pipeline",
        "judge_model": judge_label,
        "golden_dataset": str(golden_path),
        "threshold": threshold,
        "n_cases": len(goldens),
    }

    # 4. EVALUATE
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters=hyperparameters,
        async_config=AsyncConfig(run_async=False),
        error_config=ErrorConfig(ignore_errors=True),
    )
    summary = summarize_by_metric(result)
    save_results_json(summary, component="rag_pipeline_triad", hyperparameters=hyperparameters)
    return summary


def run_local(
    category: Optional[str] = None,
    threshold: float = THRESHOLD,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    skip_negative: bool = True,
) -> dict:
    """Standalone convenience: build pipeline and run evaluation."""
    pipeline = RagPipeline()
    return run(
        pipeline,
        category=category,
        threshold=threshold,
        judge_model=judge_model,
        skip_negative=skip_negative,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline-level evaluation of RAG application (Triad metrics: Contextual Relevancy, Faithfulness, Answer Relevancy)."
    )
    parser.add_argument(
        "--category",
        help="Only run test cases for a specific category.",
    )
    parser.add_argument(
        "--include-negative",
        action="store_true",
        help="Include out-of-domain negative test cases.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Pass threshold for triad metrics (default: {THRESHOLD}).",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Judge model to use (groq or model identifier).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = run_local(
        category=args.category,
        threshold=args.threshold,
        judge_model=args.judge_model,
        skip_negative=not args.include_negative,
    )
    print_summary("rag_pipeline (RAG Triad)", summary)
