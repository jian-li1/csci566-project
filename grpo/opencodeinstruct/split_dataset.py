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

def normalize_input(text):
    """Normalize whitespace for deduplication."""
    return re.sub(r"\s+", " ", text).strip()


def input_hash(text):
    """Generate stable SHA256 hash."""
    return hashlib.sha256(
        normalize_input(text).encode("utf-8")
    ).hexdigest()


def parse_list(value):
    """
    Parse a serialized list safely.

    Supports Python list representations and JSON.
    Does not execute code.
    """
    if isinstance(value, list):
        return value

    if not isinstance(value, str):
        return None

    try:
        result = ast.literal_eval(value)
    except (SyntaxError, ValueError, TypeError):
        return None

    return result if isinstance(result, list) else None


def is_valid(example, require_all_tests_pass):
    """
    Validate dataset fields and reported test results.

    Does NOT execute generated code.
    """

    problem = example.get("input")
    solution = example.get("output")

    # Require valid problem
    if not isinstance(problem, str) or not problem.strip():
        return False, "invalid_input"

    # Require valid solution
    if not isinstance(solution, str) or not solution.strip():
        return False, "invalid_output"

    # Require nonempty unit tests
    tests = parse_list(example.get("unit_tests"))

    if not tests:
        return False, "invalid_tests"

    if not all(
        isinstance(test, str) and test.strip()
        for test in tests
    ):
        return False, "invalid_tests"

    # Check that unit tests contain syntactically valid Python.
    try:
        parsed_tests = [ast.parse(test) for test in tests]
    except (SyntaxError, ValueError, TypeError):
        return False, "invalid_test_syntax"

    # Require at least one assertion in the tests.
    if not any(
        isinstance(node, ast.Assert)
        for tree in parsed_tests
        for node in ast.walk(tree)
    ):
        return False, "missing_assertions"

    if require_all_tests_pass:

        # Check reported score
        score = example.get("average_test_score")

        try:
            if abs(float(score) - 1.0) > 1e-9:
                return False, "imperfect_test_score"
        except (TypeError, ValueError):
            return False, "invalid_test_score"

        # Check reported execution statuses
        statuses = parse_list(
            example.get("tests_execution_status")
        )

        if not statuses:
            return False, "invalid_test_status"

        # Require one status per test
        if len(statuses) != len(tests):
            return False, "status_count_mismatch"

        if not all(
            isinstance(status, str)
            and status.strip().lower() == "pass"
            for status in statuses
        ):
            return False, "failed_test_status"

    return True, None


def save_jsonl(path, examples):
    """Save examples in JSONL format."""

    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )


def verify_splits(splits, sample_size, train_size, val_size, test_size):
    """Verify sizes and absence of exact duplicate problems."""

    expected = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }

    all_hashes = set()

    for name, examples in splits.items():

        assert len(examples) == expected[name], (
            f"Incorrect {name} size: "
            f"expected {expected[name]}, got {len(examples)}"
        )

        for example in examples:

            key = input_hash(example["input"])

            assert key not in all_hashes, (
                f"Duplicate problem found in {name}"
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
            "from NVIDIA OpenCodeInstruct."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="nvidia/OpenCodeInstruct",
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
        default=Path("data/opencodeinstruct"),
        help="Directory to save train.jsonl, val.jsonl, test.jsonl, and metadata.json",
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
            "Defaults to the latest revision."
        ),
    )

    parser.add_argument(
        "--require-all-tests-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require NVIDIA's reported test score to be 1.0 "
            "and all test execution statuses to indicate success. "
            "Disable with --no-require-all-tests-pass."
        ),
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100_000,
        help="Print progress every N processed examples",
    )

    args = parser.parse_args()

    # Validate arguments
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

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Initialize deterministic random generator
    rng = random.Random(args.seed)

    # Resolve dataset revision
    revision = args.revision or HfApi().dataset_info(
        args.dataset_name
    ).sha

    print("=" * 60)
    print("OpenCodeInstruct Dataset Preparation")
    print("=" * 60)

    print(f"Dataset: {args.dataset_name}")
    print(f"Source split: {args.dataset_split}")
    print(f"Revision: {revision}")
    print(f"Seed: {args.seed}")
    print(f"Sample size: {args.sample_size:,}")
    print(f"Train size: {args.train_size:,}")
    print(f"Validation size: {args.val_size:,}")
    print(f"Test size: {args.test_size:,}")
    print(f"Require all tests pass: {args.require_all_tests_pass}")
    print(f"Output directory: {output_dir}")

    # Stream full dataset
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

        # Validate metadata and test results
        valid, reason = is_valid(
            example,
            require_all_tests_pass=args.require_all_tests_pass,
        )

        if not valid:
            stats[reason] += 1
            continue

        # Deduplicate
        key = input_hash(example["input"])

        if key in seen_hashes:
            stats["duplicates"] += 1
            continue

        seen_hashes.add(key)

        valid_unique += 1
        stats["valid_unique"] += 1

        # Uniform reservoir sampling
        if len(reservoir) < args.sample_size:

            reservoir.append(example)

        else:

            j = rng.randrange(valid_unique)

            if j < args.sample_size:
                reservoir[j] = example

        # Progress logging
        if stats["processed"] % args.log_interval == 0:
            print(
                f"Processed: {stats['processed']:,} | "
                f"Valid unique: {valid_unique:,} | "
                f"Sampled: {len(reservoir):,}",
                flush=True,
            )

    # Check sufficient examples
    if len(reservoir) < args.sample_size:

        raise RuntimeError(
            f"Only {len(reservoir):,} valid unique examples found. "
            f"Need {args.sample_size:,}."
        )

    print("\nSampling complete!")

    # Shuffle sampled examples
    rng.shuffle(reservoir)

    # Split into train/val/test
    train_end = args.train_size
    val_end = train_end + args.val_size

    train = reservoir[:train_end]
    val = reservoir[train_end:val_end]
    test = reservoir[val_end:]

    splits = {
        "train": train,
        "val": val,
        "test": test,
    }

    # Verify splits
    verify_splits(
        splits,
        sample_size=args.sample_size,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    # Save JSONL files
    print("\nSaving splits...")

    for name, examples in splits.items():

        path = output_dir / f"{name}.jsonl"

        save_jsonl(path, examples)

        print(
            f"{name}: {len(examples):,} examples -> {path}"
        )

    # Save reproducibility metadata
    metadata = {
        "dataset": args.dataset_name,
        "dataset_split": args.dataset_split,
        "revision": revision,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "sampling_method": "reservoir_sampling",
        "validation_method": "reported_test_results",
        "require_all_tests_pass": args.require_all_tests_pass,
        "deduplication": "sha256_normalized_input",
        "log_interval": args.log_interval,
        "statistics": dict(stats),
        "split_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "output_directory": str(output_dir),
    }

    metadata_path = output_dir / "metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")

    print("\nAll checks passed!")

    print("\nStatistics:")
    print(json.dumps(dict(stats), indent=2))

    print("\nDone!")


if __name__ == "__main__":
    main()
