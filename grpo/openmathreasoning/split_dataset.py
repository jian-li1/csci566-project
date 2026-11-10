import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


# ==========================================
# Helper functions
# ==========================================

def normalize_problem(text):
    """Normalize whitespace for deduplication."""
    return re.sub(r"\s+", " ", text).strip()


def problem_hash(text):
    """Generate stable SHA256 hash."""
    return hashlib.sha256(
        normalize_problem(text).encode("utf-8")
    ).hexdigest()


def is_valid(example):
    """
    Validate an OpenMathReasoning example.

    Keeps only examples satisfying:
      - nonempty problem
      - nonempty generated solution
      - nonempty extracted expected answer
      - problem_type == "has_answer_extracted"
      - problem_source != "MATH_training_set"
    """

    problem = example.get("problem")
    solution = example.get("generated_solution")
    expected_answer = example.get("expected_answer")
    problem_type = example.get("problem_type")
    problem_source = example.get("problem_source")

    # --------------------------------------
    # Problem
    # --------------------------------------

    if not isinstance(problem, str) or not problem.strip():
        return False, "invalid_problem"

    # --------------------------------------
    # Generated solution
    # --------------------------------------

    if not isinstance(solution, str) or not solution.strip():
        return False, "invalid_solution"

    # --------------------------------------
    # Keep only extracted-answer problems
    # --------------------------------------

    if problem_type != "has_answer_extracted":
        return False, "not_has_answer_extracted"

    # --------------------------------------
    # Remove MATH training-set examples
    # --------------------------------------

    if problem_source == "MATH_training_set":
        return False, "math_training_set"

    # --------------------------------------
    # Expected answer
    # --------------------------------------

    if (
        not isinstance(expected_answer, str)
        or not expected_answer.strip()
    ):
        return False, "invalid_expected_answer"

    return True, None


def save_jsonl(path, examples):
    """Save examples in JSONL format."""

    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(
                json.dumps(example, ensure_ascii=False)
                + "\n"
            )


def verify_splits(
    splits,
    sample_size,
    train_size,
    val_size,
    test_size,
):
    """
    Verify split sizes and absence of exact duplicate
    normalized problems across all splits.
    """

    expected = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }

    all_hashes = set()

    for name, examples in splits.items():

        assert len(examples) == expected[name], (
            f"Incorrect {name} size: "
            f"expected {expected[name]}, "
            f"got {len(examples)}"
        )

        for example in examples:

            key = problem_hash(
                example["problem"]
            )

            assert key not in all_hashes, (
                f"Duplicate problem found in {name}"
            )

            all_hashes.add(key)

            assert (
                example["problem_type"]
                == "has_answer_extracted"
            ), (
                f"Invalid problem_type found in {name}"
            )

            assert (
                example["problem_source"]
                != "MATH_training_set"
            ), (
                f"MATH_training_set example found in {name}"
            )

    assert len(all_hashes) == sample_size, (
        f"Expected {sample_size} unique examples, "
        f"got {len(all_hashes)}"
    )


# ==========================================
# Argument parser
# ==========================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible train/val/test splits "
            "from NVIDIA OpenMathReasoning."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="nvidia/OpenMathReasoning",
        help="Hugging Face dataset name",
    )

    parser.add_argument(
        "--dataset-split",
        type=str,
        default="cot",
        help="Source dataset split",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/openmathreasoning"),
        help=(
            "Directory to save train.jsonl, val.jsonl, "
            "test.jsonl, and metadata.json"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        help="Total number of unique problems to sample",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction of sampled problems used for training",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Fraction of sampled problems used for validation",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Fraction of sampled problems used for testing",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help=(
            "Hugging Face dataset revision or commit hash. "
            "Defaults to the current revision."
        ),
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100_000,
        help="Print progress every N processed rows",
    )

    args = parser.parse_args()

    # --------------------------------------
    # Validate arguments
    # --------------------------------------

    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")

    if any(
        ratio < 0.0
        for ratio in (
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )
    ):
        parser.error("Split ratios cannot be negative")

    ratio_sum = (
        args.train_ratio
        + args.val_ratio
        + args.test_ratio
    )

    if abs(ratio_sum - 1.0) > 1e-9:
        parser.error(
            "--train-ratio + --val-ratio + --test-ratio "
            "must equal 1.0"
        )

    if args.log_interval <= 0:
        parser.error("--log-interval must be positive")

    return args


# ==========================================
# Main pipeline
# ==========================================

def main():

    args = parse_args()

    # --------------------------------------
    # Output directory
    # --------------------------------------

    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------
    # RNG
    # --------------------------------------

    rng = random.Random(args.seed)

    # --------------------------------------
    # Compute split sizes
    # --------------------------------------

    train_size = int(
        args.sample_size
        * args.train_ratio
    )

    val_size = int(
        args.sample_size
        * args.val_ratio
    )

    # Put any rounding remainder in test.
    test_size = (
        args.sample_size
        - train_size
        - val_size
    )

    # --------------------------------------
    # Resolve dataset revision
    # --------------------------------------

    revision = (
        args.revision
        or HfApi().dataset_info(
            args.dataset_name
        ).sha
    )

    print("=" * 60)
    print("OpenMathReasoning Dataset Preparation")
    print("=" * 60)

    print(f"Dataset: {args.dataset_name}")
    print(f"Source split: {args.dataset_split}")
    print(f"Revision: {revision}")
    print(f"Seed: {args.seed}")
    print(f"Sample size: {args.sample_size:,}")
    print(f"Train size: {train_size:,}")
    print(f"Validation size: {val_size:,}")
    print(f"Test size: {test_size:,}")
    print(
        "Required problem type: "
        "has_answer_extracted"
    )
    print(
        "Excluded problem source: "
        "MATH_training_set"
    )
    print(f"Output directory: {output_dir}")

    # --------------------------------------
    # Stream dataset
    # --------------------------------------

    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        streaming=True,
        revision=revision,
    )

    reservoir = []
    seen_hashes = set()
    stats = Counter()

    valid_unique = 0

    print("\nProcessing dataset...\n")

    for example in dataset:

        stats["processed"] += 1

        # ----------------------------------
        # Validation / filtering
        # ----------------------------------

        valid, reason = is_valid(example)

        if not valid:
            stats[reason] += 1
            continue

        # ----------------------------------
        # Problem-level deduplication
        # ----------------------------------

        key = problem_hash(
            example["problem"]
        )

        if key in seen_hashes:
            stats["duplicate_problem_rows"] += 1
            continue

        seen_hashes.add(key)

        valid_unique += 1
        stats["valid_unique"] += 1

        # ----------------------------------
        # Uniform reservoir sampling
        # ----------------------------------

        if len(reservoir) < args.sample_size:

            reservoir.append(example)

        else:

            j = rng.randrange(valid_unique)

            if j < args.sample_size:
                reservoir[j] = example

        # ----------------------------------
        # Progress logging
        # ----------------------------------

        if (
            stats["processed"]
            % args.log_interval
            == 0
        ):
            print(
                f"Processed: "
                f"{stats['processed']:,} | "
                f"Valid unique: "
                f"{valid_unique:,} | "
                f"Sampled: "
                f"{len(reservoir):,}",
                flush=True,
            )

    # --------------------------------------
    # Verify enough examples
    # --------------------------------------

    if len(reservoir) < args.sample_size:

        raise RuntimeError(
            f"Only {len(reservoir):,} valid unique "
            f"problems found. "
            f"Need {args.sample_size:,}."
        )

    print("\nSampling complete!")

    # --------------------------------------
    # Final deterministic shuffle
    # --------------------------------------

    rng.shuffle(reservoir)

    # --------------------------------------
    # Train / val / test split
    # --------------------------------------

    train_end = train_size

    val_end = (
        train_end
        + val_size
    )

    train = reservoir[
        :train_end
    ]

    val = reservoir[
        train_end:val_end
    ]

    test = reservoir[
        val_end:
    ]

    splits = {
        "train": train,
        "val": val,
        "test": test,
    }

    # --------------------------------------
    # Verify splits
    # --------------------------------------

    verify_splits(
        splits,
        sample_size=args.sample_size,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    # --------------------------------------
    # Save JSONL
    # --------------------------------------

    print("\nSaving splits...")

    for name, examples in splits.items():

        path = (
            output_dir
            / f"{name}.jsonl"
        )

        save_jsonl(
            path,
            examples,
        )

        print(
            f"{name}: "
            f"{len(examples):,} examples "
            f"-> {path}"
        )

    # --------------------------------------
    # Save metadata
    # --------------------------------------

    metadata = {
        "dataset": args.dataset_name,
        "dataset_split": args.dataset_split,
        "revision": revision,
        "seed": args.seed,
        "sample_size": args.sample_size,

        "sampling_method": (
            "unique_problem_reservoir_sampling"
        ),

        "filters": {
            "required_problem_type": (
                "has_answer_extracted"
            ),
            "excluded_problem_source": (
                "MATH_training_set"
            ),
        },

        "deduplication": (
            "sha256_normalized_problem"
        ),

        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },

        "split_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },

        "log_interval": (
            args.log_interval
        ),

        "statistics": dict(stats),

        "output_directory": (
            str(output_dir)
        ),
    }

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"\nMetadata saved to: "
        f"{metadata_path}"
    )

    print("\nAll checks passed!")

    print("\nStatistics:")

    print(
        json.dumps(
            dict(stats),
            indent=2,
        )
    )

    print("\nDone!")


if __name__ == "__main__":
    main()