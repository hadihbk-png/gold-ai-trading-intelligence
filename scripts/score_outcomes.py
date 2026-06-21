"""
Score pending APEX Metals AI predictions against realized settled closes.

Usage (from repo root):
    python scripts/score_outcomes.py
    python scripts/score_outcomes.py --path data/track_record.jsonl

Exits 1 if chain integrity fails after scoring.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.track_logger import LocalJsonStore
from src.score_outcomes import score_all_pending

DEFAULT_PATH = "data/track_record.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score pending APEX Metals AI predictions"
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help=f"Path to JSONL track record (default: {DEFAULT_PATH})",
    )
    args = parser.parse_args()

    store = LocalJsonStore(args.path)
    summary = score_all_pending(store)

    print("\nOutcome scoring summary")
    print("-" * 36)
    width = max(len(k) for k in summary)
    for k, v in summary.items():
        print(f"  {k:<{width}}  {v}")
    print()

    if not summary["chain_verified"]:
        print("ERROR: chain integrity check FAILED after scoring.", file=sys.stderr)
        sys.exit(1)

    print("Chain verified OK")


if __name__ == "__main__":
    main()
