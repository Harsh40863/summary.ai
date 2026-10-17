"""
Main CLI entrypoint for running the evaluation suite.
Usage:
    python -m eval
    python -m eval --suite triad
    python -m eval --suite retriever
    python -m eval --suite langsmith
    python -m eval --mode quick
"""
from eval.run_evals import main

if __name__ == "__main__":
    main()
