#!/usr/bin/env python3
"""
Convenience CLI for running RAG pipeline evaluation (RAG Triad).
Usage:
    python eval_rag_pipeline.py
    python eval_rag_pipeline.py --category financial_report
    python eval_rag_pipeline.py --threshold 0.75
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.eval_rag_pipeline import _parse_args, print_summary, run_local

if __name__ == "__main__":
    args = _parse_args()
    summary = run_local(
        category=args.category,
        threshold=args.threshold,
        judge_model=args.judge_model,
        skip_negative=not args.include_negative,
    )
    print_summary("rag_pipeline (RAG Triad)", summary)
