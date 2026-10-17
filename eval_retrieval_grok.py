#!/usr/bin/env python3
"""
Convenience CLI for running retrieval evaluation with Grok LLM Judge.
Usage:
    python eval_retrieval_grok.py
    python eval_retrieval_grok.py --category financial_report
    python eval_retrieval_grok.py --threshold 0.6
"""

from eval.retriever.eval_grok import main

if __name__ == "__main__":
    main()
