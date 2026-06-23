"""
Score pending APEX Metals AI predictions against realized settled closes.

Usage (from repo root):
    python scripts/score_outcomes.py
    python scripts/score_outcomes.py --path data/track_record.jsonl
    python scripts/score_outcomes.py --store sheets

Exits 1 if chain integrity fails after scoring.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.track_logger import LocalJsonStore, Store
from src.score_outcomes import score_all_pending

DEFAULT_PATH = "data/track_record.jsonl"


def build_store(store_kind: str, path: str) -> Store:
    """Return the store selected by --store.

    "local"  → LocalJsonStore(path) (default; nothing changes silently).
    "sheets" → the live Google Sheets store the deployed app uses, built from
               Streamlit secrets. The gspread/streamlit import stays lazy so the
               default local path never requires those packages.
    """
    if store_kind == "sheets":
        from src.track_store_factory import make_sheets_store_from_secrets
        return make_sheets_store_from_secrets()
    return LocalJsonStore(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score pending APEX Metals AI predictions"
    )
    parser.add_argument(
        "--store",
        choices=["local", "sheets"],
        default="local",
        help="Which store to score: 'local' JSONL file (default) or the live "
             "'sheets' Google Sheets ledger built from Streamlit secrets.",
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help=f"Path to JSONL track record, used only with --store local "
             f"(default: {DEFAULT_PATH})",
    )
    args = parser.parse_args()

    store = build_store(args.store, args.path)
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
