"""Shared helpers for summary.ai deepeval runs."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepeval.evaluate.types import EvaluationResult

EVAL_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_ROOT / "results"


def _ideal_answer_from_case(test_case: dict) -> str:
    chunks = test_case.get("expected_chunks") or []
    high = [c["text"] for c in chunks if c.get("relevance") == "high"]
    if high:
        return " ".join(high)
    if chunks:
        return " ".join(c["text"] for c in chunks)
    return (
        "No relevant information exists in the document corpus for this query."
    )


def load_goldens(path: str | Path) -> List[Dict[str, Any]]:
    """Load golden retriever cases and normalize fields for deepeval."""
    golden_path = Path(path)
    if not golden_path.is_absolute():
        candidates = [
            golden_path,
            Path.cwd() / golden_path,
            EVAL_ROOT / golden_path,
            EVAL_ROOT / "datasets" / golden_path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                golden_path = candidate
                break
        else:
            golden_path = EVAL_ROOT / "datasets" / golden_path.name

    with golden_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    goldens: List[Dict[str, Any]] = []
    for case in payload.get("test_cases", []):
        goldens.append(
            {
                "id": case.get("id"),
                "category": case.get("category"),
                "query": case["query"],
                "ideal_answer": _ideal_answer_from_case(case),
                "input_documents": case.get("input_documents", []),
                "expected_chunks": case.get("expected_chunks", []),
                "top_k": case.get("top_k", 3),
                "expected_threshold": case.get("expected_threshold", 0.35),
                "min_relevant_in_top_k": case.get("min_relevant_in_top_k", 1),
            }
        )
    return goldens


def summarize_by_metric(result: EvaluationResult) -> Dict[str, Any]:
    """Aggregate per-metric scores and pass rates from a deepeval run."""
    scores: Dict[str, List[float]] = defaultdict(list)
    passes: Dict[str, List[bool]] = defaultdict(list)
    thresholds: Dict[str, float] = {}

    for test_result in result.test_results:
        if not test_result.metrics_data:
            continue
        for metric in test_result.metrics_data:
            if metric.score is not None:
                scores[metric.name].append(float(metric.score))
            passes[metric.name].append(bool(metric.success))
            if metric.threshold is not None:
                thresholds[metric.name] = float(metric.threshold)

    summary: Dict[str, Any] = {"metrics": {}, "test_run_id": result.test_run_id}
    for name, values in scores.items():
        avg = sum(values) / len(values) if values else 0.0
        pass_rate = (
            sum(1 for p in passes[name] if p) / len(passes[name])
            if passes[name]
            else 0.0
        )
        summary["metrics"][name] = {
            "average_score": round(avg, 4),
            "pass_rate": round(pass_rate, 4),
            "threshold": thresholds.get(name),
            "n_cases": len(values),
        }
    return summary


def save_results_json(
    summary: Dict[str, Any],
    *,
    component: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{component}_results_{stamp}.json"
    payload = {
        "component": component,
        "hyperparameters": hyperparameters or {},
        "summary": summary,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def print_summary(component: str, summary: Dict[str, Any]) -> None:
    print()
    print("=" * 70)
    print(f"  summary.ai — {component} evaluation")
    print("=" * 70)
    metrics = summary.get("metrics") or {}
    if not metrics:
        print("  (no metric aggregates returned)")
        print("=" * 70)
        return

    print(f"{'Metric':<28}{'Avg Score':>12}{'Pass Rate':>12}{'Threshold':>12}")
    print("-" * 70)
    for name, data in metrics.items():
        threshold = data.get("threshold")
        threshold_str = f"{threshold:.2f}" if threshold is not None else "—"
        print(
            f"{name:<28}"
            f"{data['average_score']:>12.4f}"
            f"{data['pass_rate']:>12.1%}"
            f"{threshold_str:>12}"
        )
    print("=" * 70)
