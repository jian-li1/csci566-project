from pathlib import Path
from datasets import load_dataset

def custom_dataset_loader(
    file_path: str,
    qCol: str = "problem",
    aCol: str = "solution",
    sCol: str = None
):
    file = Path(file_path).expanduser()

    dataset = load_dataset(
        "json",
        data_files=str(file),
        split="train"
    )

    required = [qCol, aCol]
    if sCol:
        required.append(sCol)

    missing = [
        col for col in required
        if col not in dataset.column_names
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    def normalize(example):
        return {
            "system": (example[sCol] or "") if sCol else "",
            "question": example[qCol],
            "answer": example[aCol]
        }

    return dataset.map(
        normalize,
        remove_columns=dataset.column_names
    )