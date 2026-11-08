import argparse
import ast
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

def normalize_question(text):
    """Normalize whitespace for deduplication."""
    return re.sub(r"\s+", " ", text).strip()


def question_hash(text):
    """Generate stable SHA256 hash."""
    return hashlib.sha256(
        normalize_question(text).encode("utf-8")
    ).hexdigest()


def contains_assert(tree):
    """Return True if a Python AST contains an assert statement."""
    return any(
        isinstance(node, ast.Assert)
        for node in ast.walk(tree)
    )


def is_valid(example, require_valid_tests):
    """
    Validate a KodCode example.

    Does NOT execute generated code.

    Checks:
      - nonempty question
      - nonempty solution
      - nonempty test
      - syntactically valid Python test code
      - at least one assertion (optional)
      - at least one successful GPT verification trial
    """

    question = example.get("question")
    solution = example.get("solution")
    test = example.get("test")

    # --------------------------------------
    # Question
    # --------------------------------------

    if not isinstance(question, str) or not question.strip():
        return False, "invalid_question"

    # --------------------------------------
    # Solution
    # --------------------------------------

    if not isinstance(solution, str) or not solution.strip():
        return False, "invalid_solution"

    # --------------------------------------
    # Test
    # --------------------------------------

    if not isinstance(test, str) or not test.strip():
        return False, "invalid_test"

    # Check test syntax.
    try:
        test_tree = ast.parse(test)
    except (SyntaxError, ValueError, TypeError):
        return False, "invalid_test_syntax"

    if require_valid_tests:

        # Require at least one assertion.
        if not contains_assert(test_tree):
            return False, "missing_assertions"

    # --------------------------------------
    # GPT verification metadata
    # --------------------------------------

    pass_sequence = example.get("gpt_pass_sequence")

    if not isinstance(pass_sequence, list):
        return False, "invalid_pass_sequence"

    if len(pass_sequence) == 0:
        return False, "empty_pass_sequence"

    # Validate sequence contents.
    if not all(
        value in (0, 1)
        for value in pass_sequence
    ):
        return False, "invalid_pass_sequence"

    # KodCode's published solution should correspond to
    # a successful verification trial. Require evidence
    # that at least one trial passed.
    if not any(value == 1 for value in pass_sequence):
        return False, "no_successful_trial"

    # --------------------------------------
    # Pass percentage sanity check
    # --------------------------------------

    pass_percentage = example.get("gpt_pass_percentage")

    try:
        pass_percentage = float(pass_percentage)
    except (TypeError, ValueError):
        return False, "invalid_pass_percentage"

    if not 0.0 <= pass_percentage <= 1.0:
        return False, "invalid_pass_percentage"

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
    normalized questions.
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

            key = question_hash(example["question"])

            assert key not in all_hashes, (
                f"Duplicate question found in {name}"
            )

            all_hashes.add(key)

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
            "from KodCode-V1."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="KodCode/KodCode-V1",
        help="Hugging Face dataset name",
    )

    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Source dataset split",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/kodcode"),
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
        help="Total number of examples to sample",
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=7_000,
        help="Number of training examples",
    )

    parser.add_argument(
        "--val-size",
        type=int,
        default=2_000,
        help="Number of validation examples",
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=1_000,
        help="Number of testing examples",
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
        "--require-valid-tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require test code to contain at least one "
            "Python assert statement. Disable with "
            "--no-require-valid-tests."
        ),
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10_000,
        help="Print progress every N processed examples",
    )

    args = parser.parse_args()

    # --------------------------------------
    # Validate arguments
    # --------------------------------------

    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")

    if any(
        size < 0
        for size in (
            args.train_size,
            args.val_size,
            args.test_size,
        )
    ):
        parser.error("Split sizes cannot be negative")

    if (
        args.train_size
        + args.val_size
        + args.test_size
        != args.sample_size
    ):
        parser.error(
            "--train-size + --val-size + --test-size "
            "must equal --sample-size"
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
    # Resolve dataset revision
    # --------------------------------------

    revision = (
        args.revision
        or HfApi().dataset_info(
            args.dataset_name
        ).sha
    )

    print("=" * 60)
    print("KodCode-V1 Dataset Preparation")
    print("=" * 60)

    print(f"Dataset: {args.dataset_name}")
    print(f"Source split: {args.dataset_split}")
    print(f"Revision: {revision}")
    print(f"Seed: {args.seed}")
    print(f"Sample size: {args.sample_size:,}")
    print(f"Train size: {args.train_size:,}")
    print(f"Validation size: {args.val_size:,}")
    print(f"Test size: {args.test_size:,}")
    print(
        f"Require valid tests: "
        f"{args.require_valid_tests}"
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
        # Validation
        # ----------------------------------

        valid, reason = is_valid(
            example,
            require_valid_tests=args.require_valid_tests,
        )

        if not valid:
            stats[reason] += 1
            continue

        # ----------------------------------
        # Deduplication
        # ----------------------------------

        key = question_hash(
            example["question"]
        )

        if key in seen_hashes:
            stats["duplicates"] += 1
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
            f"examples found. "
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

    train_end = args.train_size

    val_end = (
        train_end
        + args.val_size
    )

    train = reservoir[:train_end]

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
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
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
            "reservoir_sampling"
        ),

        "validation_method": (
            "kodcode_reported_verification_metadata"
        ),

        "require_valid_tests": (
            args.require_valid_tests
        ),

        "deduplication": (
            "sha256_normalized_question"
        ),

        "log_interval": (
            args.log_interval
        ),

        "statistics": dict(stats),

        "split_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },

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