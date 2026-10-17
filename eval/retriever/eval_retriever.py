"""
DeepEval retriever evaluation: contextual precision and recall on the golden set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional

from dotenv import load_dotenv

_EVAL_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _EVAL_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from eval.harness import (
    load_goldens,
    print_summary,
    save_results_json,
    summarize_by_metric,
)
from eval.retriever.retriever_wrapper import (
    AppDocumentRetriever,
    retriever_for_golden,
)

load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT / "backend" / ".env")

GOLDEN_PATH = _EVAL_ROOT / "datasets" / "retriever_golden_dataset.json"
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o-mini")
GEMINI_JUDGE_MODEL = os.getenv("DEEPEVAL_GEMINI_JUDGE_MODEL", "gemini-2.0-flash")
THRESHOLD = float(os.getenv("DEEPEVAL_RETRIEVER_THRESHOLD", "0.7"))


from deepeval.models.base_model import DeepEvalBaseLLM


class GrokDeepEvalJudge(DeepEvalBaseLLM):
    """DeepEval LLM Judge wrapper for xAI / Grok models."""

    def __init__(
        self,
        model_name: str = "grok-2-latest",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = (
            api_key
            or os.getenv("XAI_API_KEY")
            or os.getenv("GROK_API_KEY")
        )
        self.base_url = base_url or os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        super().__init__(model_name=model_name)

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


def _resolve_judge_model(judge_model: str):
    """Use Grok, OpenAI, or Gemini based on configuration and model name."""
    xai_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    if judge_model.startswith("grok") or (xai_key and not os.getenv("OPENAI_API_KEY")):
        grok_name = judge_model if judge_model.startswith("grok") else "grok-2-latest"
        return GrokDeepEvalJudge(model_name=grok_name, api_key=xai_key)

    if os.getenv("OPENAI_API_KEY"):
        return judge_model

    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if gemini_key:
        gemini_name = judge_model
        if gemini_name.startswith("gemini/"):
            gemini_name = gemini_name.split("/", 1)[1]
        elif judge_model.startswith("gpt-"):
            gemini_name = GEMINI_JUDGE_MODEL
        return GeminiModel(model=gemini_name, api_key=gemini_key)
    return judge_model


def _chunk_size_sentences() -> int:
    return int(os.getenv("CHUNK_SIZE_SENTENCES", "2"))


def _build_test_cases(
    goldens: list,
    retriever_builder: Callable,
) -> List[LLMTestCase]:
    test_cases: List[LLMTestCase] = []
    for golden in goldens:
        retriever = retriever_builder(golden)
        retrieved = retriever.invoke(golden["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=golden["query"],
                expected_output=golden["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )
    return test_cases


def run(
    retriever_builder: Callable = retriever_for_golden,
    *,
    golden_path: Path = GOLDEN_PATH,
    threshold: float = THRESHOLD,
    judge_model: str = JUDGE_MODEL,
    category: Optional[str] = None,
    skip_negative: bool = False,
) -> dict:
    goldens = load_goldens(golden_path)

    if category:
        goldens = [g for g in goldens if g.get("category") == category]
    if skip_negative:
        goldens = [g for g in goldens if g.get("category") != "negative_test"]

    if not goldens:
        raise ValueError("No golden test cases matched the selected filters.")

    test_cases = _build_test_cases(goldens, retriever_builder)
    judge = _resolve_judge_model(judge_model)
    judge_label = getattr(judge, "model", judge_model)

    metrics = [
        ContextualRecallMetric(
            threshold=threshold,
            model=judge,
            include_reason=True,
        ),
        ContextualPrecisionMetric(
            threshold=threshold,
            model=judge,
            include_reason=True,
        ),
    ]

    hyperparameters = {
        "retriever": "app_document_mmr_faiss",
        "embedding_model": os.getenv(
            "ENABLE_SENTENCE_TRANSFORMERS", "hashing_fallback"
        ),
        "chunk_size_sentences": _chunk_size_sentences(),
        "judge_model": judge_label,
        "golden_set": str(golden_path),
        "metric_threshold": threshold,
        "n_cases": len(goldens),
    }

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters=hyperparameters,
    )
    summary = summarize_by_metric(result)
    save_results_json(summary, component="retriever", hyperparameters=hyperparameters)
    return summary


def run_local() -> dict:
    """Standalone convenience: default app retriever factory per golden."""
    return run(retriever_for_golden)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deepeval contextual precision/recall on the retriever."
    )
    parser.add_argument(
        "--category",
        help="Only run cases with this category label.",
    )
    parser.add_argument(
        "--skip-negative",
        action="store_true",
        help="Skip negative / out-of-domain test cases.",
    )
    parser.add_argument(
        "--precision-threshold",
        type=float,
        default=THRESHOLD,
        help="Pass threshold for both precision and recall metrics.",
    )
    parser.add_argument(
        "--recall-threshold",
        type=float,
        default=None,
        help="Optional recall threshold (defaults to --precision-threshold).",
    )
    parser.add_argument(
        "--judge-model",
        default=JUDGE_MODEL,
        help="OpenAI-compatible judge model for deepeval metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    metric_threshold = args.precision_threshold
    if args.recall_threshold is not None and args.recall_threshold != metric_threshold:
        print(
            "Note: deepeval applies one threshold per metric instance; "
            "using --precision-threshold for both metrics."
        )
    summary = run(
        retriever_for_golden,
        threshold=metric_threshold,
        judge_model=args.judge_model,
        category=args.category,
        skip_negative=args.skip_negative,
    )
    print_summary("retriever", summary)
