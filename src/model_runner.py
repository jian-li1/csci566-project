"""SFT runner: automatic train/validation split or supplied train/val JSONL pair.

Validation is always required. Full dataset is used by default.
"""

import argparse
import os
from torch.utils.data import Subset

from custom_dataset_loader import custom_dataset_loader
from run_paths import DEFAULT_RUNS_ROOT, format_lr, make_run_dir
from run_summary import RunSummary
from transformermodel import TransformerModel
from training_loops import TrainingLoops


def describe_filter(stats):
    """One summary line from TransformerModel.filter_dataset stats."""
    line = (
        f"kept {stats['kept']}/{stats['total']} | "
        f"too long {stats['too_long']} | empty {stats['empty']}"
    )
    if "p50" in stats:
        line += f" | tokens p50={stats['p50']} p95={stats['p95']} max={stats['max']}"
    return line


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
    # Results go to <runs-root>/<dataset-name>/<model>/<run name>/ unless --outdir is given
    parser.add_argument("--outdir", default=None, help="Run folder; default builds one under --runs-root")
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT, help="Top folder for runs and index.csv")
    parser.add_argument("--dataset-name", default=None, help="Dataset folder name under --runs-root; default is taken from the data path")
    parser.add_argument("--tag", default=None, help="Optional label added to the run name")
    parser.add_argument("--max-len", type=int, default=4096, help="Max tokens per sample; longer samples are skipped")
    parser.add_argument("--log-every", type=int, default=None, help="Print a progress line every N optimizer steps; default is about 20 lines per epoch")
    args = parser.parse_args()

    if args.samples is not None and args.samples <= 0:
        parser.error("--samples must be positive")
    if args.max_len <= 0:
        parser.error("--max-len must be positive")
    if args.log_every is not None and args.log_every <= 0:
        parser.error("--log-every must be positive")
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

    # The tokenizer is needed to measure samples, so build the model first
    model = TransformerModel(args.model, max_length=args.max_len)

    # Skip samples longer than --max-len once, before training starts
    if dataset is not None:
        dataset = model.filter_dataset(dataset, "dataset")
        print(f"Automatic split input: {len(dataset)} records")
    else:
        trainset = model.filter_dataset(trainset, "train")
        valset = model.filter_dataset(valset, "val")
        print(f"Provided split: train={len(trainset)}, val={len(valset)} records")

    if args.dataset_name is None:
        if args.source == "hf":
            args.dataset_name = "hf_math"
        else:
            data_file = args.dataset_file or args.train_file
            args.dataset_name = os.path.basename(os.path.dirname(os.path.abspath(data_file)))
    if args.outdir is None:
        args.outdir = make_run_dir(
            args.runs_root, args.dataset_name, args.model,
            args.ep, args.lr, args.samples, args.tag
        )
    print(f"Run folder: {args.outdir}")

    summary = RunSummary(
        args.outdir,
        index_path=os.path.join(args.runs_root, "index.csv"),
        index_info={
            "dataset": args.dataset_name,
            "model": model.model_name,
            "lr": format_lr(args.lr),
            "samples": args.samples or "full",
        },
    )
    summary.sections["Run"]["Run folder"] = os.path.abspath(args.outdir)
    lora = model.lora_config
    # peft expands "all-linear" into the matched module names
    targets = lora.target_modules
    if not isinstance(targets, str):
        targets = ",".join(sorted(targets))
    summary.set_section("Model", {
        "Model": f"{model.model_name} ({model.model_id})",
        "LoRA": (
            f"r={lora.r}, alpha={lora.lora_alpha}, dropout={lora.lora_dropout}, "
            f"targets={targets}"
        ),
        "Trainable": (
            f"{model.trainable_params / 1e6:.1f}M / {model.total_params / 1e9:.2f}B params "
            f"({100 * model.trainable_params / model.total_params:.2f}%)"
        ),
    })

    data = {"Source": args.source}
    if args.source == "hf":
        data["Dataset"] = "Hugging Face math (hf_math_loader)"
    elif args.dataset_file is not None:
        data["Dataset file"] = args.dataset_file
    else:
        data["Train file"] = args.train_file
        data["Val file"] = args.val_file
    if dataset is not None:
        data["Split"] = f"automatic {args.tp:g}/{args.vp:g} (seed 42)"
    data["Samples cap"] = f"{args.samples} (debug subset)" if args.samples else "none (full dataset)"
    data["Max length"] = f"{args.max_len} tokens"
    for name, stats in model.filter_stats.items():
        data[name.capitalize()] = describe_filter(stats)
    summary.set_section("Data", data)

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
        summary=summary,
        log_every=args.log_every,
    )
    training.train()


if __name__ == "__main__":
    main()
