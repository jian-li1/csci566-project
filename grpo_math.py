"""
GRPO training script (TRL) for the Competition Math dataset.

Ablation design:
  - Hold rollouts/optimizer-step fixed at 64 for every arm.
  - Vary only --G (completions per prompt): 4, 8, or 16.
  - Stop by --max-steps so compute matches across arms.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from competition_math import (
    MATH_VERIFY_AVAILABLE,
    CompetitionMathEnv,
    extract_boxed_answer,
)

# ---------------------------------------------------------------------------
# Fixed experiment constants (same for every G arm)
# ---------------------------------------------------------------------------
# Completions used in one optimizer update. Must stay constant across --G.
ROLLOUTS_PER_STEP = 64
# How those 64 are built: microbatch × grad_accum (not via generation_batch_size).
MICRO_BATCH_COMPLETIONS = 8  # per_device_train_batch_size
GRAD_ACCUM_STEPS = 8  # 8 * 8 = 64 completions per optimizer step

DEFAULT_DATASET = "qwedsacf/competition_math"
DEFAULT_MAX_STEPS = 400
DEFAULT_SEED = 42
DEFAULT_LR = 2e-5

# Pin RL sampling/KL; TRL defaults have changed between versions.
BETA = 0.001  # KL coefficient vs reference policy
TEMPERATURE = 1.0
WARMUP_RATIO = 0.03

# Qwen3.5 is loaded as a text-only causal LM by Transformers, while vLLM uses
# the checkpoint's multimodal wrapper and therefore expects different prefixes.
QWEN35_MODEL_TYPES = ("qwen3_5", "qwen3_5_text")
QWEN35_VLLM_PREFIX_MAP = (
    ("lm_head.", "language_model.lm_head."),
    ("model.", "language_model.model."),
)

# Columns the binary correctness reward reads from the train dataset.
REQUIRED_COLUMNS = ("prompt", "problem", "solution")


# ---------------------------------------------------------------------------
# CLI — sweep knobs live here; fixed geometry stays in constants above
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL GRPO on Competition Math + LoRA")
    parser.add_argument(
        "--G",
        type=int,
        required=True,
        choices=(4, 8, 16),
        help="Group size (completions per prompt). Rollouts/step stay fixed at 64.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Hugging Face model id or local base-model path.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=(
            "A local .parquet / .json / .jsonl path, a directory containing split "
            "files, or a Hugging Face dataset id (default: qwedsacf/competition_math)."
        ),
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Split name for a Hub dataset or local split directory (default: train).",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--init-adapter",
        type=str,
        default=None,
        help="Path to an SFT LoRA adapter to continue from (SFT→GRPO arm).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--report-to",
        type=str,
        default="wandb",
        choices=("wandb", "none"),
        help="Trainer reporting integration (default: wandb).",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Use transformers generation instead of colocated vLLM.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading — normalize Competition Math rows to reward columns
# ---------------------------------------------------------------------------
def _normalize_row(row: dict) -> dict:
    """
    Map one raw row → {prompt, problem, solution, level, type, ...}.

    The reference solution stays in the row for the reward function, but only the
    plain-string prompt is tokenized and shown to the model.
    """
    out = dict(row)
    missing = [column for column in ("problem", "solution") if not out.get(column)]
    if missing:
        raise ValueError(f"Row missing required values {missing}. Keys={sorted(out)}")

    env = CompetitionMathEnv()
    env.load_problem(
        problem=out["problem"],
        solution=out["solution"],
        level=out.get("level"),
        problem_type=out.get("type"),
    )
    out["prompt"] = env.prompt
    return out


def _has_usable_reference(row: dict) -> bool:
    """Reject known-bad rows whose worked solution has no non-empty boxed answer."""
    answer = extract_boxed_answer(row.get("solution", ""))
    return answer is not None and bool(answer.strip())


def _load_local_dataset(path: Path, split: str) -> Dataset:
    """Load one local file, or ``<directory>/<split>.<supported extension>``."""
    if path.is_dir():
        candidates = [
            path / f"{split}.parquet",
            path / f"{split}.jsonl",
            path / f"{split}.json",
        ]
        matches = [candidate for candidate in candidates if candidate.exists()]
        if not matches:
            raise ValueError(
                f"No '{split}.parquet', '{split}.jsonl', or '{split}.json' in {path}"
            )
        path = matches[0]

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path), split="train")
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as dataset_file:
            records = [json.loads(line) for line in dataset_file if line.strip()]
        return Dataset.from_list(records)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as dataset_file:
            records = json.load(dataset_file)
        if isinstance(records, dict):
            records = [records]
        return Dataset.from_list(records)
    raise ValueError(
        f"Unsupported local file type '{suffix}'. Use .parquet, .json, or .jsonl."
    )


def load_train_dataset(source: str, split: str) -> Dataset:
    """Load data from disk or the HF Hub, then normalize + validate schema."""
    path = Path(source)

    # --- resolve source ---
    if path.exists():
        dataset = _load_local_dataset(path, split)
    else:
        # Not a local path → treat as Hugging Face dataset id.
        dataset = load_dataset(source, split=split)

    # --- pick split if a custom loader returned a DatasetDict ---
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Split '{split}' not in dataset. Available: {list(dataset)}")
        dataset = dataset[split]

    # Two rows in the 12,500-row source currently end in ``\boxed{}``.
    # They cannot support binary correctness, so report and remove them explicitly.
    rows_before_filter = len(dataset)
    dataset = dataset.filter(_has_usable_reference)
    dropped_rows = rows_before_filter - len(dataset)
    if dropped_rows:
        print(f"Dropped {dropped_rows} rows with missing/empty boxed gold answers")

    # --- normalize rows, then fail fast on missing reward columns ---
    dataset = dataset.map(_normalize_row)
    missing = [column for column in REQUIRED_COLUMNS if column not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Dataset missing required columns {missing}. Have {dataset.column_names}. "
            "Competition Math rows need problem and solution."
        )

    # This path deliberately uses plain strings instead of chat message lists.
    assert len(dataset) > 0, "Train dataset is empty"
    assert isinstance(dataset[0]["prompt"], str), (
        "prompt must be a plain string; message lists make TRL apply a chat "
        f"template. Got {type(dataset[0]['prompt'])}"
    )
    return dataset


# ---------------------------------------------------------------------------
# Token accounting — matched rollouts ≠ matched tokens across G arms
# ---------------------------------------------------------------------------
class CompletionTokenCallback(TrainerCallback):
    """
    Accumulate generated completion tokens from TRL's completions/mean_length.

    step_tokens ≈ mean_length × rollouts_per_step
    Use total_completion_tokens (not just steps) when plotting accuracy vs compute.
    """

    def __init__(self, rollouts_per_step: int, output_dir: str | Path):
        self.rollouts_per_step = rollouts_per_step
        self.output_dir = Path(output_dir)
        self.total_completion_tokens = 0.0
        self.history: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        mean_len = logs.get("completions/mean_length")
        if mean_len is None:
            return

        # Approximate tokens generated this optimizer step (64 rollouts).
        step_tokens = float(mean_len) * self.rollouts_per_step
        self.total_completion_tokens += step_tokens
        row = {
            "step": int(state.global_step),
            "mean_completion_length": float(mean_len),
            "step_tokens": step_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
        self.history.append(row)

        # on_log runs after the trainer already flushed `logs`; push derived
        # totals explicitly so W&B can plot accuracy vs tokens later.
        if state.is_world_process_zero:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "completions/step_tokens": step_tokens,
                            "completions/total_tokens": self.total_completion_tokens,
                        },
                        step=state.global_step,
                    )
            except Exception:  # noqa: BLE001
                pass

    def on_train_end(self, args, state, control, **kwargs):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "completion_tokens.json"
        payload = {
            "rollouts_per_step": self.rollouts_per_step,
            "total_completion_tokens": self.total_completion_tokens,
            "num_logged_steps": len(self.history),
            "history": self.history,
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"Wrote token accounting → {out_path} "
            f"(total_completion_tokens={self.total_completion_tokens:.0f})"
        )


# ---------------------------------------------------------------------------
# Reward — TRL calls this on generated completions; returns one float each
# ---------------------------------------------------------------------------
def make_competition_math_reward():
    """Build a GRPO reward_fn that scores binary correctness with the environment."""
    env = CompetitionMathEnv()

    def reward_competition_math(
        completions, problem, solution, level=None, type=None, **kwargs
    ):
        # Dataset columns are repeated to match completions by TRL.
        levels = level if level is not None else [None] * len(completions)
        problem_types = type if type is not None else [None] * len(completions)
        rewards = []
        for completion, question, gold, difficulty, subject in zip(
            completions, problem, solution, levels, problem_types
        ):
            # TRL may pass plain strings or chat-style message lists.
            if isinstance(completion, list):
                text = completion[-1].get("content", "") if completion else ""
            else:
                text = completion or ""

            # Bind the matching dataset row; never expose the worked solution.
            env.load_problem(question, gold, difficulty, subject)
            out = env.step(text)
            reward = out["reward"]
            # TRL expects a float per completion; unscored → 0.0.
            rewards.append(0.0 if reward is None else float(reward))
        return rewards

    return reward_competition_math


# ---------------------------------------------------------------------------
# Model — base model + LoRA (fresh or continued from an SFT adapter)
# ---------------------------------------------------------------------------
def build_lora_config() -> LoraConfig:
    """Build a broadly compatible causal-LM LoRA configuration."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def load_model(model_id: str, init_adapter: str | None):
    # No device_map: let the Trainer / vLLM colocate placement own the device.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    if init_adapter:
        # SFT → GRPO arm: continue training an existing adapter.
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        # Fresh LoRA for GRPO-from-base.
        model = get_peft_model(model, build_lora_config())

    model.print_trainable_parameters()
    return model


def patch_qwen35_vllm_weight_sync(trainer: GRPOTrainer, model: PeftModel) -> bool:
    """Map text-only Qwen3.5 weight names to vLLM's multimodal wrapper."""
    vllm_generation = getattr(trainer, "vllm_generation", None)
    model_type = getattr(model.get_base_model().config, "model_type", None)
    if vllm_generation is None or model_type not in QWEN35_MODEL_TYPES:
        return False

    original_fix = vllm_generation._fix_param_name_to_vllm

    def fix_param_name(name: str, extra_prefixes: list[str] | None = None) -> str:
        name = original_fix(name, extra_prefixes)
        for prefix, replacement in QWEN35_VLLM_PREFIX_MAP:
            if name.startswith(prefix):
                return name.replace(prefix, replacement, 1)
        return name

    vllm_generation._fix_param_name_to_vllm = fix_param_name
    return True


# ---------------------------------------------------------------------------
# Main — wire data, model, config, trainer; run GRPO
# ---------------------------------------------------------------------------
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for GRPO training, but PyTorch cannot see a GPU. "
            "Run this command inside a GPU-enabled allocation."
        )
    print(f"Training device: cuda:0 ({torch.cuda.get_device_name(0)})")

    args = parse_args()

    if not MATH_VERIFY_AVAILABLE:
        raise ImportError(
            "GRPO training requires math-verify for reproducible symbolic grading. "
            "Install it with: pip install 'math-verify[antlr4_13_2]'"
        )

    # --- sanity checks on fixed batch geometry ---
    if ROLLOUTS_PER_STEP % args.G != 0:
        raise ValueError(
            f"ROLLOUTS_PER_STEP ({ROLLOUTS_PER_STEP}) must be divisible by G ({args.G})"
        )
    if MICRO_BATCH_COMPLETIONS * GRAD_ACCUM_STEPS != ROLLOUTS_PER_STEP:
        raise ValueError("Microbatch × grad accum must equal ROLLOUTS_PER_STEP")

    # Prompts per optimizer step changes with G; rollouts (64) do not.
    prompts_per_step = ROLLOUTS_PER_STEP // args.G
    lr_tag = f"{args.lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    run_name = f"math-grpo-G{args.G}-lr{lr_tag}-s{args.seed}"
    output_dir = args.output_dir or f"GRPO/{run_name}"

    print(
        f"{run_name} | rollouts/step={ROLLOUTS_PER_STEP} | "
        f"prompts/step={prompts_per_step} | max_steps={args.max_steps} | "
        f"init_adapter={args.init_adapter!r}"
    )
    print(
        "Batch geometry: "
        f"per_device_train_batch_size={MICRO_BATCH_COMPLETIONS} × "
        f"gradient_accumulation_steps={GRAD_ACCUM_STEPS} = "
        f"{ROLLOUTS_PER_STEP} completions/optimizer step (all G)."
    )

    # --- data / tokenizer / model ---
    dataset = load_train_dataset(args.dataset, args.dataset_split)
    print(dataset)
    print(f"columns={dataset.column_names} | rows={len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model_id, args.init_adapter)

    # --- GRPO hyperparameters ---
    # Important: do NOT set generation_batch_size. That would generate 64 once and
    # then split them across several optimizer steps (stale rollouts + unequal compute).
    # With the defaults, each optimizer step generates a fresh set of 64 completions.
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=args.lr,
        # Fixed geometry across G arms:
        per_device_train_batch_size=MICRO_BATCH_COMPLETIONS,  # 8
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,  # 8 → 64 / step
        num_generations=args.G,  # only ablation knob in the batch math
        max_completion_length=1024,  # room for <think>…<answer>
        optim="adamw_8bit",
        max_steps=args.max_steps,  # matched compute; not num_train_epochs
        warmup_steps=max(1, round(WARMUP_RATIO * args.max_steps)),
        bf16=True,
        beta=BETA,
        temperature=TEMPERATURE,
        seed=args.seed,
        data_seed=args.seed,
        gradient_checkpointing=True,  # needed at long completion length
        save_strategy="steps",
        save_steps=50,
        save_total_limit=None,  # keep early ckpts for accuracy-vs-compute curves
        logging_steps=1,
        log_completions=True,  # inspect samples in W&B (catch reward hacking)
        report_to=[] if args.report_to == "none" else [args.report_to],
        remove_unused_columns=False,  # keep problem/solution for the reward fn
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",  # vLLM shares the training GPU
        vllm_gpu_memory_utilization=0.5,  # leave headroom for the train model
        vllm_max_model_length=2048,
        run_name=run_name,
    )

    # --- train ---
    # Loop inside TRL: sample G completions/prompt → score → group-normalize → update.
    token_cb = CompletionTokenCallback(ROLLOUTS_PER_STEP, output_dir)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_competition_math_reward()],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[token_cb],
    )

    if patch_qwen35_vllm_weight_sync(trainer, model):
        print("Enabled Qwen3.5 Hugging Face → vLLM weight-name compatibility mapping.")

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
