"""Run folder layout: <runs_root>/<dataset>/<model>/<run name>/

The run name is built from the start time and the main settings, e.g.
20261028-1124_ep3_lr1e-4_n50_j13389883. model_runner.py and
scripts/train.sh both get the folder from here, so they always agree.

Command line (prints the created folder):
    python src/run_paths.py --dataset competition_math --model qwen --ep 3 --lr 1e-4
"""

import argparse
import os
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNS_ROOT = os.path.join(REPO_ROOT, "runs")


def format_lr(lr):
    """1e-4 -> "1e-4", 2.5e-5 -> "2.5e-5" (shorter than Python's 0.0001)."""
    mantissa, exponent = f"{lr:e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def run_name(epochs, lr, samples=None, tag=None, job_id=None):
    parts = [
        datetime.now().strftime("%Y%m%d-%H%M"),
        f"ep{epochs}",
        f"lr{format_lr(lr)}",
        f"n{samples}" if samples else "full",
    ]
    if tag:
        parts.append(tag)
    if job_id:
        parts.append(f"j{job_id}")
    return "_".join(parts)


def make_run_dir(runs_root, dataset, model, epochs, lr, samples=None, tag=None):
    """Create and return a new, empty run folder."""
    name = run_name(epochs, lr, samples, tag, os.environ.get("SLURM_JOB_ID"))
    base = os.path.join(runs_root, dataset, model.lower(), name)

    # Two local runs started in the same minute would share a name
    path = base
    suffix = 2
    while os.path.exists(path):
        path = f"{base}-{suffix}"
        suffix += 1
    os.makedirs(path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Create a run folder and print its path")
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    print(make_run_dir(
        args.runs_root, args.dataset, args.model,
        args.ep, args.lr, args.samples, args.tag
    ))


if __name__ == "__main__":
    main()
