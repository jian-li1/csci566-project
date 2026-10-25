"""Gemma SFT runner: automatic train/validation split or supplied train/val JSONL pair.

Validation is always required. Full dataset is used by default.
"""

import argparse
from torch.utils.data import Subset

from custom_dataset_loader import custom_dataset_loader
from transformermodel import TransformerModel
from training_loops import TrainingLoops

def main():
    parser = argparse.ArgumentParser(description="Train Gemma with required validation")
    parser.add_argument("--source", choices=["custom", "hf"], default="custom")

    # Option A: one JSONL file; the runner splits it into train and val.
    parser.add_argument("--dataset-file", default=None, help="One JSONL file to split into train and validation")
    # Option B: both explicit train and val files must be supplied together.
    parser.add_argument("--train-file", default=None, help="Pre-split training JSONL file")
    parser.add_argument("--val-file", default=None, help="Pre-split validation JSONL file")
    parser.add_argument("--qcol", default="problem", help="Source question column")
    parser.add_argument("--acol", default="solution", help="Source answer column")
    parser.add_argument("--scol", default=None, help="Source system column")
    parser.add_argument("--samples", type=int, default=None, help="Optional debugging cap; default None uses every row")
    parser.add_argument("--ep", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gstep", type=int, default=16)
    parser.add_argument("--bsize", type=int, default=1)
    parser.add_argument("--tp", type=float, default=0.8)
    parser.add_argument("--vp", type=float, default=0.2)
    parser.add_argument("--model", type=str, default= "llama", help="Model use Llama/Llama-3.2-3B by default")
    parser.add_argument("--outdir", default="./checkpoints")
    args = parser.parse_args()

    if args.samples is not None and args.samples <= 0:
        parser.error("--samples must be positive")
    if args.bsize <= 0 or args.gstep <= 0 or args.ep <= 0:
        parser.error("--bsize, --gstep, and --ep must all be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")

    if args.source == "custom":
        single_file = args.dataset_file is not None
        paired_files = args.train_file is not None or args.val_file is not None
        if single_file == paired_files:
            parser.error(
                "For --source custom, choose exactly one: "
                "--dataset-file FILE (auto split), OR "
                "--train-file TRAIN --val-file VAL (provided split)."
            )
        if paired_files and not (args.train_file and args.val_file):
            parser.error("--train-file and --val-file are BOTH required for a provided split")
    else:
        if args.dataset_file or args.train_file or args.val_file:
            parser.error("For --source hf, do not supply any JSONL file arguments")

    # Auto-split always needs a valid train/validation ratio.
    auto_split = args.source == "hf" or args.dataset_file is not None

    if auto_split and not (0 < args.tp < 1 and 0 < args.vp < 1 and abs(args.tp + args.vp - 1.0) <= 1e-6):
        parser.error("For automatic splitting, --tp and --vp must be positive and sum to 1")

    dataset = None
    trainset = None
    valset = None

    if args.source == "hf":
        from hf_math_loader import MathDataset
        print("Loading full Hugging Face math dataset for automatic splitting")
        dataset = MathDataset()
    elif args.dataset_file is not None:
        dataset = custom_dataset_loader(
            args.dataset_file,
            qCol=args.qcol,
            aCol=args.acol,
            sCol=args.scol
        )
    else:
        trainset = custom_dataset_loader(
            args.train_file,
            qCol=args.qcol,
            aCol=args.acol,
            sCol=args.scol
        )

        valset = custom_dataset_loader(
            args.val_file,
            qCol=args.qcol,
            aCol=args.acol,
            sCol=args.scol
        )

    # No cap by default. If debugging, cap only training candidates, not
    # the separately supplied validation dataset.
    if args.samples is not None:
        if dataset is not None:
            dataset = Subset(dataset, range(min(args.samples, len(dataset))))
        else:
            trainset = Subset(trainset, range(min(args.samples, len(trainset))))
        print("WARNING: Debugging subset enabled: not using the full training dataset")

    if dataset is not None:
        print(f"Automatic split input: {len(dataset)} records")
    else:
        print(f"Provided split: train={len(trainset)}, val={len(valset)} records")

    model = TransformerModel(args.model)
    training = TrainingLoops(
        model=model,
        dataset=dataset,
        trainset=trainset,
        valset=valset,
        training_portion=args.tp,
        validation_portion=args.vp,
        batch_size=args.bsize,
        gradient_accumulation_steps=args.gstep,
        epochs=args.ep,
        lr=args.lr,
        output_dir=args.outdir,
    )
    training.train()


if __name__ == "__main__":
    main()
