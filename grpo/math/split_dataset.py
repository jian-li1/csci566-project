import argparse
import json
from pathlib import Path

from datasets import load_dataset, concatenate_datasets

DATASET_NAME = "qwedsacf/competition_math"
SEED = 42
OUTPUT_DIR = Path("data/competition_math")

parser = argparse.ArgumentParser(
    description="Create deterministic train, validation, and test splits."
)
parser.add_argument(
    "--sample-size",
    type=int,
    default=None,
    help="Number of examples to sample before splitting (default: use all examples).",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=OUTPUT_DIR,
    help=f"Directory for the generated JSONL files (default: {OUTPUT_DIR}).",
)
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)

# Load dataset
dataset = load_dataset(DATASET_NAME)

# Combine existing splits before creating our own
full_dataset = concatenate_datasets(list(dataset.values()))

# Shuffle deterministically
full_dataset = full_dataset.shuffle(seed=SEED)

# Optionally select a deterministic random sample
if args.sample_size is not None:
    if args.sample_size < 4:
        parser.error("--sample-size must be at least 4 to create all three splits")
    if args.sample_size > len(full_dataset):
        parser.error(
            f"--sample-size cannot exceed the dataset size ({len(full_dataset)})"
        )
    full_dataset = full_dataset.select(range(args.sample_size))

# Split: 70% train, 20% val, 10% test
train_test = full_dataset.train_test_split(
    test_size=0.30,
    seed=SEED,
)

val_test = train_test["test"].train_test_split(
    test_size=1/3,
    seed=SEED,
)

splits = {
    "train": train_test["train"],
    "val": val_test["train"],
    "test": val_test["test"],
}

# Save as JSONL
for name, split in splits.items():
    path = args.output_dir / f"{name}.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for example in split:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"{name}: {len(split)} examples -> {path}")

print("\nDone!")
