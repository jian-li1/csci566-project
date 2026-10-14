"""Load a JSONL training file into the format expected by TrainingLoops."""

from pathlib import Path
from datasets import load_dataset


def custom_dataset_loader(
    file_path: str,
    qCol: str = "problem",
    aCol: str = "solution",
):
    """
    Load all records from a JSONL file.

    Example row: {"problem": "...", "solution": "...", "level": "Level 2"}
    Output row:  {"question": "...", "answer": "..."}
    """
    file = Path(file_path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"JSONL dataset not found: {file}")
    if qCol == aCol:
        raise ValueError("Question and answer must use different source columns.")

    dataset = load_dataset("json", data_files=str(file), split="train")
    missing = [name for name in (qCol, aCol) if name not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {file}. "
            f"Available columns: {dataset.column_names}"
        )

    # Discard unrelated columns and standardize names for the training loop.
    dataset = dataset.select_columns([qCol, aCol])
    rename = {}
    if qCol != "question":
        rename[qCol] = "question"
    if aCol != "answer":
        rename[aCol] = "answer"
    if rename:
        dataset = dataset.rename_columns(rename)

    if len(dataset) == 0:
        raise ValueError(f"JSONL file contains no training examples: {file}")
    return dataset
