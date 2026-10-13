from datasets import load_dataset

def custom_dataset_loader(type: str):
    dataset = load_dataset(
        "json",
        data_files=f"./data/{type}.jsonl",
        split="train"
    )

    dataset = dataset.select_columns(["problem", "solution"])

    dataset = dataset.rename_columns({
        "problem": "question",
        "solution": "answer"
    })

    return dataset