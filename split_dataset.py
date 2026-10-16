import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

DATASET_NAME = "qwedsacf/competition_math"
SEED = 42
OUTPUT_DIR = Path("data")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
dataset = load_dataset(DATASET_NAME)

# Combine existing splits before creating our own
full_dataset = concatenate_datasets(list(dataset.values()))

# Shuffle deterministically
full_dataset = full_dataset.shuffle(seed=SEED)

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
    path = OUTPUT_DIR / f"{name}.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for example in split:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"{name}: {len(split)} examples -> {path}")

print("\nDone!")
