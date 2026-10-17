"""
Retrieval eval: CONTEXTUAL PRECISION & RECALL (with Grok as LLM Judge)

Precision and Recall measure whether the RAG application's retriever
successfully selects relevant context while filtering out noise.

We measure:
    - Contextual Precision: Proportion of retrieved document chunks that are
      genuinely relevant to the query (signal vs noise).
    - Contextual Recall: Proportion of golden reference facts / expected chunks
      that were successfully returned by the retriever (completeness / coverage).
    - F1 Score: Harmonic mean of precision and recall.
    - Hit Rate / Pass Rate: Percentage of queries meeting the threshold.

Evaluated against the golden dataset (eval/datasets/retriever_golden_dataset.json)
for summary.ai's DocumentSearchEngine (FAISS + MMR retrieval), using Grok
(xAI OpenAI-compatible API) as the LLM judge with automatic retry and backoff.
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

# Ensure summary.ai project root & backend are in sys.path
_EVAL_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _EVAL_DIR.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"

for _p in (_PROJECT_ROOT, _BACKEND_DIR, _EVAL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Load environment variables
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_BACKEND_DIR / ".env")

# Ensure NLTK data path points to local project data
import nltk
_local_nltk = _PROJECT_ROOT / ".nltk_data"
if _local_nltk.is_dir() and str(_local_nltk) not in nltk.data.path:
    nltk.data.path.insert(0, str(_local_nltk))

from openai import OpenAI
from eval.harness import load_goldens
from eval.retriever.retriever_wrapper import AppDocumentRetriever, retriever_for_golden


# ============================================================
# 2. CONFIG
# ============================================================
DEFAULT_GOLDEN_PATH = _EVAL_DIR / "datasets" / "retriever_golden_dataset.json"
RESULTS_DIR = _EVAL_DIR / "results"

# Grok (xAI) / Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")

if XAI_API_KEY:
    DEFAULT_PROVIDER = "grok"
    XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    GROK_MODEL = os.getenv("GROK_MODEL", os.getenv("XAI_MODEL", "grok-2-latest"))
elif GROQ_API_KEY:
    DEFAULT_PROVIDER = "groq"
    XAI_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROK_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
else:
    DEFAULT_PROVIDER = "grok"
    XAI_BASE_URL = "https://api.x.ai/v1"
    GROK_MODEL = "grok-2-latest"

# Evaluation Hyperparameters
DEFAULT_THRESHOLD = float(os.getenv("RETRIEVAL_PASS_THRESHOLD", "0.7"))
MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "5"))
BACKOFF_BASE_S = float(os.getenv("EVAL_BACKOFF_BASE_S", "2.5"))


# ============================================================
# 3. RETRIEVAL METRICS TRACKER
# ============================================================
class RetrievalMetrics:
    """
    Tracks contextual precision, contextual recall, latency,
    API retries, and pass rates across retrieval benchmark runs.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.retries = 0
        self.test_cases: List[Dict[str, Any]] = []

    def record_case(self, case_result: Dict[str, Any]):
        self.test_cases.append(case_result)

    @property
    def total_cases(self) -> int:
        return len(self.test_cases)

    @property
    def avg_precision(self) -> float:
        if not self.test_cases:
            return 0.0
        return sum(c["precision"] for c in self.test_cases) / len(self.test_cases)

    @property
    def avg_recall(self) -> float:
        if not self.test_cases:
            return 0.0
        return sum(c["recall"] for c in self.test_cases) / len(self.test_cases)

    @property
    def avg_f1(self) -> float:
        p = self.avg_precision
        r = self.avg_recall
        if (p + r) == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.test_cases if c["passed"])

    @property
    def pass_rate(self) -> float:
        if not self.test_cases:
            return 0.0
        return 100 * self.pass_count / len(self.test_cases)


# ============================================================
# 4. RETRY WRAPPER & GROK LLM JUDGE
# ============================================================
def call_with_retries(fn, metrics: RetrievalMetrics):
    """
    Executes an LLM or retrieval call with exponential backoff on errors.
    """
    metrics.calls += 1

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = fn()
            metrics.successes += 1
            return result
        except Exception as e:
            if attempt < MAX_RETRIES:
                metrics.retries += 1
                backoff = BACKOFF_BASE_S * (2 ** attempt)
                time.sleep(backoff)
            else:
                metrics.failures += 1
                print(f"  [!] Evaluation call failed after {MAX_RETRIES} retries: {e}")
                return None


class GrokJudge:
    """
    Uses Grok (xAI) via OpenAI-compatible endpoint to evaluate
    Contextual Precision and Contextual Recall.
    Falls back to deterministic overlap matching if no API key is provided.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GROK_MODEL,
        base_url: str = XAI_BASE_URL,
    ):
        self.api_key = api_key or XAI_API_KEY or GROQ_API_KEY
        self.model = model
        self.base_url = base_url
        self.client: Optional[OpenAI] = None

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            print("[INFO] No XAI_API_KEY or GROQ_API_KEY found. Running with deterministic ground-truth matcher.")
            print("       To use Grok / Groq live, add XAI_API_KEY=... or GROQ_API_KEY=... in your .env file.\n")

    def _judge_with_grok(
        self,
        query: str,
        retrieved_chunks: List[str],
        expected_chunks: List[str],
        ideal_answer: str,
        category: str,
    ) -> Dict[str, Any]:
        """Calls Grok to evaluate precision and recall."""
        assert self.client is not None

        chunks_text = "\n---\n".join(
            f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)
        ) if retrieved_chunks else "(No chunks retrieved)"

        expected_text = "\n".join(
            f"- {fact}" for fact in expected_chunks
        ) if expected_chunks else ideal_answer

        system_prompt = (
            "You are an expert AI evaluator judging the retrieval stage of a RAG pipeline.\n"
            "Evaluate Contextual Precision and Contextual Recall for the retrieved document chunks.\n\n"
            "DEFINITIONS:\n"
            "1. CONTEXTUAL PRECISION (0.0 to 1.0):\n"
            "   - What fraction of the retrieved chunks contain relevant information that directly answers\n"
            "     or provides useful context for the user's query?\n"
            "   - If no chunks were retrieved:\n"
            "     * Score 1.0 if the query is out-of-domain/negative (no relevant information was expected).\n"
            "     * Score 0.0 if the query expected relevant information.\n"
            "   - If chunks were retrieved for a negative query with no relevant info, precision is 0.0.\n"
            "   - Otherwise: score = (count of relevant chunks) / (total retrieved chunks).\n\n"
            "2. CONTEXTUAL RECALL (0.0 to 1.0):\n"
            "   - Did the retrieved chunks cover all the key facts/information from the ground truth reference?\n"
            "   - If the query is an out-of-domain/negative query where no information exists:\n"
            "     * Score 1.0 if the retriever returned no relevant chunks (correctly did not fabricate).\n"
            "     * Score 0.0 if the retriever returned chunks falsely claimed to answer it.\n"
            "   - Otherwise: score = (count of ground-truth facts present in retrieved chunks) / (total ground-truth facts).\n\n"
            "Respond ONLY with a valid JSON object in this exact schema:\n"
            "{\n"
            '  "precision": <float between 0.0 and 1.0>,\n'
            '  "precision_reason": "<1-2 sentence explanation>",\n'
            '  "recall": <float between 0.0 and 1.0>,\n'
            '  "recall_reason": "<1-2 sentence explanation>"\n'
            "}"
        )

        user_prompt = (
            f"User Query:\n{query}\n\n"
            f"Category: {category}\n\n"
            f"Ground Truth Reference Facts:\n{expected_text}\n\n"
            f"Retrieved Chunks ({len(retrieved_chunks)} total):\n{chunks_text}\n"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        time.sleep(1.0)

        raw = response.choices[0].message.content or "{}"
        clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        clean = re.sub(r"```$", "", clean.strip(), flags=re.MULTILINE)

        data = json.loads(clean.strip())
        precision = float(data.get("precision", 0.0))
        recall = float(data.get("recall", 0.0))
        precision_reason = data.get("precision_reason", "")
        recall_reason = data.get("recall_reason", "")

        return {
            "precision": max(0.0, min(1.0, precision)),
            "recall": max(0.0, min(1.0, recall)),
            "precision_reason": precision_reason,
            "recall_reason": recall_reason,
            "judge": f"grok ({self.model})",
        }

    def _judge_deterministic(
        self,
        query: str,
        retrieved_chunks: List[str],
        expected_chunks: List[str],
        ideal_answer: str,
        category: str,
    ) -> Dict[str, Any]:
        """
        Deterministic lexical / substring overlap against golden chunks.
        Used when live Grok API key is not yet configured.
        """
        if category == "negative_test" or not expected_chunks:
            # Negative test case: expecting no relevant chunks
            precision = 1.0 if not retrieved_chunks else 0.0
            recall = 1.0
            reason = (
                "Negative test case (out-of-domain query correctly rejected)"
                if precision == 1.0
                else "Retriever returned chunk for out-of-domain query"
            )
            return {
                "precision": precision,
                "recall": recall,
                "precision_reason": reason,
                "recall_reason": reason,
                "judge": "ground-truth-match (fallback)",
            }

        # Check precision: how many retrieved chunks match expected chunks
        relevant_retrieved = 0
        for chunk in retrieved_chunks:
            chunk_clean = chunk.lower().strip()
            for exp in expected_chunks:
                exp_clean = exp.lower().strip()
                if exp_clean in chunk_clean or chunk_clean in exp_clean:
                    relevant_retrieved += 1
                    break
                else:
                    exp_words = set(exp_clean.split())
                    chunk_words = set(chunk_clean.split())
                    overlap = len(exp_words & chunk_words)
                    if overlap / max(1, len(exp_words)) >= 0.5:
                        relevant_retrieved += 1
                        break

        precision = (relevant_retrieved / len(retrieved_chunks)) if retrieved_chunks else 0.0

        # Check recall: how many expected chunks are covered by retrieved chunks
        all_retrieved_text = " ".join(retrieved_chunks).lower()
        covered_expected = 0
        for exp in expected_chunks:
            exp_clean = exp.lower().strip()
            if exp_clean in all_retrieved_text:
                covered_expected += 1
            else:
                exp_words = set(exp_clean.split())
                retrieved_words = set(all_retrieved_text.split())
                overlap = len(exp_words & retrieved_words)
                if overlap / max(1, len(exp_words)) >= 0.5:
                    covered_expected += 1

        recall = (covered_expected / len(expected_chunks)) if expected_chunks else 1.0

        return {
            "precision": max(0.0, min(1.0, precision)),
            "recall": max(0.0, min(1.0, recall)),
            "precision_reason": f"{relevant_retrieved}/{len(retrieved_chunks)} retrieved chunks matched ground truth",
            "recall_reason": f"{covered_expected}/{len(expected_chunks)} expected chunks captured in context",
            "judge": "ground-truth-match (fallback)",
        }

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[str],
        expected_chunks: List[str],
        ideal_answer: str,
        category: str,
    ) -> Dict[str, Any]:
        if self.client is not None:
            return self._judge_with_grok(
                query, retrieved_chunks, expected_chunks, ideal_answer, category
            )
        return self._judge_deterministic(
            query, retrieved_chunks, expected_chunks, ideal_answer, category
        )


# ============================================================
# 5. BENCHMARK
# ============================================================
def benchmark(
    pipeline_builder: Callable = retriever_for_golden,
    golden_path: Path = DEFAULT_GOLDEN_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    category: Optional[str] = None,
    skip_negative: bool = False,
    judge: Optional[GrokJudge] = None,
) -> RetrievalMetrics:
    if judge is None:
        judge = GrokJudge()

    goldens = load_goldens(golden_path)
    if category:
        goldens = [g for g in goldens if g.get("category") == category]
    if skip_negative:
        goldens = [g for g in goldens if g.get("category") != "negative_test"]

    if not goldens:
        raise ValueError("No golden test cases matched the selected criteria.")

    metrics = RetrievalMetrics(threshold=threshold)
    judge_name = judge.model if judge.client else "deterministic ground-truth"
    print(f"Measuring retrieval performance across {len(goldens)} golden cases...")
    print(f"Judge: Grok ({judge_name}) | Pass Threshold: {threshold:.2f}\n")

    for idx, golden in enumerate(goldens, 1):
        case_id = golden["id"]
        cat = golden.get("category", "general")
        query = golden["query"]
        expected_chunks = [c["text"] for c in golden.get("expected_chunks", []) if "text" in c]
        ideal_answer = golden.get("ideal_answer", "")

        # 1. Retrieve using summary.ai pipeline
        start_time = time.time()
        retriever = pipeline_builder(golden)
        retrieved_docs = retriever.invoke(query)
        elapsed_s = time.time() - start_time
        retrieved_chunks = [d.page_content for d in retrieved_docs]

        # 2. Evaluate with retry wrapper
        eval_data = call_with_retries(
            lambda: judge.evaluate(
                query=query,
                retrieved_chunks=retrieved_chunks,
                expected_chunks=expected_chunks,
                ideal_answer=ideal_answer,
                category=cat,
            ),
            metrics,
        )

        if eval_data is None:
            prec, rec = 0.0, 0.0
            prec_reason = "Evaluation failed after retries"
            rec_reason = "Evaluation failed after retries"
            judge_used = "error"
        else:
            prec = eval_data["precision"]
            rec = eval_data["recall"]
            prec_reason = eval_data["precision_reason"]
            rec_reason = eval_data["recall_reason"]
            judge_used = eval_data.get("judge", "grok")

        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        passed = (prec >= threshold) and (rec >= threshold)

        status_icon = "✓" if passed else "✗"
        print(
            f"  {status_icon} [{idx:02d}/{len(goldens):02d}] {case_id} ({cat}): "
            f"Precision={prec:.2f}, Recall={rec:.2f}, F1={f1:.2f} ({elapsed_s:.2f}s)"
        )

        metrics.record_case({
            "id": case_id,
            "category": cat,
            "query": query,
            "retrieved_count": len(retrieved_chunks),
            "expected_count": len(expected_chunks),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "passed": passed,
            "precision_reason": prec_reason,
            "recall_reason": rec_reason,
            "latency_s": round(elapsed_s, 3),
            "judge": judge_used,
        })

    return metrics


# ============================================================
# 6. REPORT
# ============================================================
def report(metrics: RetrievalMetrics, save_json: bool = True) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION REPORT (PRECISION & RECALL)")
    print("=" * 70)

    print(f"Total Test Cases : {metrics.total_cases}")
    print(f"Successful Evals : {metrics.successes}")
    print(f"Failed Evals     : {metrics.failures}")
    print(f"API Retries      : {metrics.retries}")

    print("-" * 70)
    print(f"Contextual Precision (avg) : {metrics.avg_precision * 100:.2f}%")
    print(f"Contextual Recall (avg)    : {metrics.avg_recall * 100:.2f}%")
    print(f"F1 Score (avg)             : {metrics.avg_f1 * 100:.2f}%")
    print(f"Pass Rate (>= {metrics.threshold:.2f})       : {metrics.pass_rate:.2f}% ({metrics.pass_count}/{metrics.total_cases})")
    print("=" * 70)

    # Per-category breakdown
    categories = sorted(set(c["category"] for c in metrics.test_cases))
    print(f"\n{'Category':<24}{'Cases':>8}{'Precision':>14}{'Recall':>12}{'Pass Rate':>12}")
    print("-" * 70)
    for cat in categories:
        cat_cases = [c for c in metrics.test_cases if c["category"] == cat]
        c_count = len(cat_cases)
        c_prec = sum(c["precision"] for c in cat_cases) / c_count
        c_rec = sum(c["recall"] for c in cat_cases) / c_count
        c_passed = 100 * sum(1 for c in cat_cases if c["passed"]) / c_count
        print(f"{cat:<24}{c_count:>8}{c_prec * 100:>13.1f}%{c_rec * 100:>11.1f}%{c_passed:>11.1f}%")
    print("=" * 70)

    summary_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cases": metrics.total_cases,
        "average_precision": round(metrics.avg_precision, 4),
        "average_recall": round(metrics.avg_recall, 4),
        "average_f1": round(metrics.avg_f1, 4),
        "pass_rate": round(metrics.pass_rate, 2),
        "threshold": metrics.threshold,
        "cases": metrics.test_cases,
    }

    if save_json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_file = RESULTS_DIR / f"retrieval_grok_results_{stamp}.json"
        out_file.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(f"\n📄 Results saved to: {out_file.relative_to(_PROJECT_ROOT)}")

    return summary_payload


# ============================================================
# 7. ENTRYPOINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Retrieval precision & recall evaluation using Grok as LLM judge."
    )
    parser.add_argument(
        "--category",
        help="Only run test cases for a specific category.",
    )
    parser.add_argument(
        "--skip-negative",
        action="store_true",
        help="Skip negative / out-of-domain test cases.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Pass threshold for precision and recall (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--model",
        default=GROK_MODEL,
        help=f"Grok model name (default: {GROK_MODEL}).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results JSON to disk.",
    )
    args = parser.parse_args()

    judge = GrokJudge(model=args.model)
    metrics = benchmark(
        threshold=args.threshold,
        category=args.category,
        skip_negative=args.skip_negative,
        judge=judge,
    )
    report(metrics, save_json=not args.no_save)


if __name__ == "__main__":
    main()
