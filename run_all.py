"""Run test evaluation for completed checkpoints. No optimizer or training stage."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluate import run_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/eval.json")
    parser.add_argument("--run", action="append", help="Only this run name; repeat to select several.")
    parser.add_argument("--limit", type=int, help="Smoke test using the first N fixed test questions.")
    parser.add_argument("--check-only", action="store_true", help="Check config/data without loading a model.")
    parser.add_argument("--stage", choices=["all", "generate", "grade", "references"], default="all")
    args = parser.parse_args()
    try:
        run_all(args.config, names=args.run, limit=args.limit, check_only=args.check_only, stage=args.stage)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        parser.exit(1, f"Evaluation stopped: {error}\n")


if __name__ == "__main__":
    main()
