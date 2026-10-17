#!/usr/bin/env python3
"""
Convenience CLI for running generator evaluation with Groq / Grok LLM Judge.
Usage:
    python eval_generator_groq.py
    python eval_generator_groq.py --provider groq --model llama-3.3-70b-versatile
    python eval_generator_groq.py --provider grok --model grok-2-latest
    python eval_generator_groq.py --category financial_report
    python eval_generator_groq.py --threshold 0.75
"""

from eval.generator.eval_generator import main

if __name__ == "__main__":
    main()
