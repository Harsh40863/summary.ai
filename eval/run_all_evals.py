#!/usr/bin/env python3
"""Convenience alias for root run_evals.py"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_evals import main

if __name__ == "__main__":
    main()
