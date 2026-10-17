"""
eval/eval_generator.py
======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

Answer Relevancy: does the generated answer directly address the user query
without irrelevant tangents or fluff?

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the golden dataset), NOT the retriever's output. So a low score
is purely the generator's fault --- the context was already correct.

Supports Groq (default), Grok (xAI), or OpenAI as the LLM judge.

Usage:
    python eval/eval_generator.py
    python -m eval.eval_generator
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT / "backend" / ".env")

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from eval.harness import (
    load_goldens,
    print_summary,
    save_results_json,
    summarize_by_metric,
)
from src.generator import generate  # your generator: generate(query, context) -> answer

GOLDEN_PATH = _PROJECT_ROOT / "eval" / "datasets" / "retriever_golden_dataset.json"
DEFAULT_JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "groq")
THRESHOLD = float(os.getenv("DEEPEVAL_GENERATOR_THRESHOLD", "0.7"))


# ============================================================
# LLM JUDGE WRAPPERS (Groq / Grok / OpenAI)
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

    def load_model(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


class GrokJudgeModel(DeepEvalBaseLLM):
    """DeepEval judge model wrapper for xAI Grok."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name or os.getenv("GROK_MODEL", "grok-2-latest")
        self.api_key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        self.base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        super().__init__(model_name=self.model_name)

    def load_model(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


def _resolve_judge(judge_name: str):
    """Select appropriate judge model (Groq, Grok, OpenAI, Gemini)."""
    # 1. Groq (if explicitly asked or if GROQ_API_KEY exists)
    if "groq" in judge_name.lower() or (
        os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY")
    ):
        model = None if "groq" in judge_name.lower() else judge_name
        return GroqJudgeModel(model_name=model)

    # 2. Grok (xAI)
    if "grok" in judge_name.lower() or os.getenv("XAI_API_KEY"):
        model = None if "grok" in judge_name.lower() else judge_name
        return GrokJudgeModel(model_name=model)

    # 3. Default OpenAI string model
    return judge_name


# ============================================================
# RUN EVALUATION IN ISOLATION
# ============================================================
def run(
    golden_path: Path = GOLDEN_PATH,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    threshold: float = THRESHOLD,
    category: Optional[str] = None,
    skip_negative: bool = True,  # Isolation tests evaluate groundedness on known-good contexts
) -> dict:
    # 1. LOAD the golden set (query + golden expected context chunks)
    goldens = load_goldens(golden_path)

    if category:
        goldens = [g for g in goldens if g.get("category") == category]
    if skip_negative:
        goldens = [g for g in goldens if g.get("category") != "negative_test"]

    if not goldens:
        raise ValueError("No golden test cases matched the selected criteria.")

    judge = _resolve_judge(judge_model)
    judge_label = getattr(judge, "model_name", str(judge_model))

    print(f"Evaluating {len(goldens)} cases with generator in ISOLATION...")
    print(f"Judge: {judge_label} | Threshold: {threshold:.2f}\n")

    # 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
    test_cases = []
    for g in goldens:
        # Extract known-good context chunks from the golden set
        context = [
            c["text"]
            for c in g.get("expected_chunks", [])
            if isinstance(c, dict) and "text" in c
        ]
        if not context:
            context = [g.get("ideal_answer", "")]

        # RUN the generator on the golden context -> actual_output
        answer = generate(g["query"], context)

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,             # the generated answer we're judging
                retrieval_context=context,        # faithfulness checks the answer against THIS
                expected_output=g.get("ideal_answer"),  # ground truth reference answer
            )
        )

    # 3. THE METRICS --- decompose actual_output into claims, attribute each to context
    metrics = [
        FaithfulnessMetric(
            threshold=threshold,
            model=judge,
            include_reason=True,   # prints WHY each score --- shows which claims were unsupported
        ),
        AnswerRelevancyMetric(
            threshold=threshold,
            model=judge,
            include_reason=True,
        ),
    ]

    hyperparameters = {
        "evaluation_type": "generator_in_isolation",
        "judge_model": judge_label,
        "golden_dataset": str(golden_path),
        "threshold": threshold,
        "n_cases": len(goldens),
    }

    # 4. EVALUATE --- runs the metrics on every case, prints a report
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters=hyperparameters,
    )
    summary = summarize_by_metric(result)
    save_results_json(summary, component="generator_isolation", hyperparameters=hyperparameters)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Component-level evaluation of the generator in isolation (DeepEval)."
    )
    parser.add_argument(
        "--category",
        help="Only evaluate test cases for a specific category.",
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
        help=f"Pass threshold for metrics (default: {THRESHOLD}).",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Judge model to use (groq, grok, or model identifier).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = run(
        threshold=args.threshold,
        judge_model=args.judge_model,
        category=args.category,
        skip_negative=not args.include_negative,
    )
    print_summary("generator (in isolation)", summary)
