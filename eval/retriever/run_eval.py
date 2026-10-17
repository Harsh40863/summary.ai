"""CLI entrypoint (see eval/README.md)."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.harness import print_summary
from eval.retriever.eval_retriever import _parse_args, retriever_for_golden, run

if __name__ == "__main__":
    args = _parse_args()
    summary = run(
        retriever_for_golden,
        threshold=args.precision_threshold,
        judge_model=args.judge_model,
        category=args.category,
        skip_negative=args.skip_negative,
    )
    print_summary("retriever", summary)
