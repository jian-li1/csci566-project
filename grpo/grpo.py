"""
GRPO training script (TRL) for multi-task runs (Reasoning Gym, math, coding stub).

Ablation design:
  - Hold rollouts/optimizer-step fixed at 64 for every arm.
  - Vary only --G (completions per prompt): 4, 8, or 16.
  - Stop by --max-steps so compute matches across arms.
  - One run = one --task.
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from compute_budget import ComputeBudgetCallback

# ---------------------------------------------------------------------------
# Imports from this repo when launched as: python train/grpo.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grpo.countdown import CountdownEnv
from grpo.knights_knaves import KnightsKnavesEnv
from grpo.shortest_path import ShortestPathEnv
from grpo.math import CompetitionMathEnv, extract_boxed_answer
from grpo.metrics import GRPOMetrics, GRPOMetricsCallback, make_env_reward_fn

# ---------------------------------------------------------------------------
# Fixed experiment constants (same for every G arm)
# ---------------------------------------------------------------------------
# Base model for all runs (Hugging Face id or local path). Edit this to switch.
MODEL_ID = "Qwen/Qwen3.5-4B-Base"
#MODEL_2_ID = "Llama/Llama-3.2-3B-Base"
#MODEL_3_ID = ""

# Completions used in one optimizer update. Must stay constant across --G.
ROLLOUTS_PER_STEP = 64
# How those 64 are built: microbatch × grad_accum (not via generation_batch_size).
MICRO_BATCH_COMPLETIONS = 8  # per_device_train_batch_size
GRAD_ACCUM_STEPS = 8  # 8 * 8 = 64 completions per optimizer step

DEFAULT_MAX_STEPS = 400
DEFAULT_SEED = 42
DEFAULT_LR = 2e-5

# Pin RL sampling/KL; TRL defaults have changed between versions.
BETA = 0.001  # KL coefficient vs reference policy
TEMPERATURE = 1.0

# Qwen3.5 is loaded as a text-only causal LM by Transformers, while vLLM uses
# the checkpoint's multimodal wrapper and therefore expects different prefixes.
QWEN35_MODEL_TYPES = ("qwen3_5", "qwen3_5_text")
QWEN35_VLLM_PREFIX_MAP = (
    ("lm_head.", "language_model.lm_head."),
    ("model.", "language_model.model."),
)

# tasks
RG_TASKS = ("countdown", "knights_knaves", "shortest_path")
TASKS = {
    "countdown": {
        "required_columns": ("prompt", "task", "entry"),
    },
    "knights_knaves": {
        "required_columns": ("prompt", "task", "entry"),
    },
    "shortest_path": {
        "required_columns": ("prompt", "task", "entry"),
    },
    "math": {
        "required_columns": ("prompt", "task", "entry"),
    },
    "coding": {
        "required_columns": ("prompt", "task", "entry"),
    },
}

# ---------------------------------------------------------------------------
# CLI — sweep knobs live here; fixed geometry stays in constants above
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL GRPO on Reasoning Gym (countdown, knights_knaves, shortest path), Math and Coding (pending)")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=("countdown", "knights_knaves", "shortest_path", "math", "coding"),
        help="Task to train on. Choices: countdown, knights_knaves, shortest_path, math, coding",
    )
    parser.add_argument(
        "--G",
        type=int,
        required=True,
        choices=(4, 8, 16),
        help="Group size (completions per prompt). Rollouts/step stay fixed at 64.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Train data: local .parquet / .json / .jsonl path, a directory of those, "
            "or a Hugging Face dataset id. Row schema depends on --task: "
            "RG tasks (countdown, knights_knaves, shortest_path) need "
            "question/answer/metadata (prompt may be built in normalize); "
            "math needs problem/solution (and optional level/type)."
        ),
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help=(
            "Compute budget in weighted tokens (FLOPs/N) for this model. "
            "Omit to track spend without stopping. Per-model B values TBD."
        ),
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="none",
        choices=("none", "sft_then_grpo:sft", "sft_then_grpo:grpo"),
        help=(
            "Budget stage: none = full budget (pure GRPO); "
            "sft_then_grpo:sft = 25%% of B; sft_then_grpo:grpo = 75%% of B."
        ),
    )

    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Split name when --dataset is a HF hub id or a DatasetDict (default: train).",
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading — bring your own file / HF id; normalize to reward columns
# ---------------------------------------------------------------------------
def _parse_meta(meta):
    """meta may be missing, a JSON string, or already a dict."""
    if meta is None:
        return {}
    if isinstance(meta, str):
        return json.loads(meta)
    if isinstance(meta, dict):
        return meta
    raise TypeError(f"Unsupported meta type: {type(meta)}")


def _rg_entry(out: dict) -> dict:
    """Build a Reasoning Gym entry; parse metadata in-place when present."""
    entry = {"question": out["question"], "answer": out["answer"]}
    if out.get("metadata") is not None:
        entry["metadata"] = _parse_meta(out["metadata"])
        out["metadata"] = entry["metadata"]
    return entry


def _has_usable_reference(row: dict) -> bool:
    """Reject math rows whose worked solution has no non-empty boxed answer."""
    answer = extract_boxed_answer(row.get("solution", ""))
    return answer is not None and bool(answer.strip())


def _normalize_row(row: dict, task: str) -> dict:
    """
    Map one raw row → {prompt, task, entry, ...}.

    Builds `prompt` via the task env; `entry` is what metrics/reward pass to
    load_problem (RG dict, or math/MBPP-compatible payload).
    """
    out = dict(row)
    out["task"] = task

    if task in RG_TASKS:
        if task == "countdown":
            env = CountdownEnv()
        elif task == "knights_knaves":
            env = KnightsKnavesEnv()
        else:
            env = ShortestPathEnv()
        entry = _rg_entry(out)
        obs = env.load_problem(entry)
        out["entry"] = entry
        out["prompt"] = obs["prompt"]
    elif task == "math":
        env = CompetitionMathEnv()
        entry = {
            "problem": out["problem"],
            "solution": out["solution"],
        }
        if out.get("level") is not None:
            entry["level"] = out["level"]
        if out.get("type") is not None:
            entry["type"] = out["type"]
        # Prefer entry dict once CompetitionMathEnv.load_problem(dict) lands;
        # positional keeps normalize working in the meantime.
        obs = env.load_problem(
            entry["problem"],
            entry["solution"],
            level=entry.get("level"),
            problem_type=entry.get("type"),
        )
        out["entry"] = entry
        out["prompt"] = obs["prompt"]
    elif task == "coding":
        # Pass-through stub — expect prompt (+ optional entry) already in the row.
        if "entry" not in out:
            out["entry"] = {}
        if "prompt" not in out:
            raise ValueError("coding pass-through requires a prebuilt 'prompt' column")
    else:
        raise ValueError(f"Unknown task: {task}")

    return out


def load_train_dataset(source: str, split: str, task: str) -> Dataset:
    """Load user data from disk or the HF Hub, then normalize + validate schema."""
    path = Path(source)

    # --- resolve source ---
    if path.exists():
        if path.is_dir():
            dataset = load_dataset("parquet", data_dir=str(path), split=split)
        else:
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                dataset = load_dataset("parquet", data_files=str(path), split="train")
            elif suffix in {".jsonl", ".json"}:
                dataset = load_dataset("json", data_files=str(path), split="train")
            else:
                raise ValueError(
                    f"Unsupported local file type '{suffix}'. Use .parquet, .json, or .jsonl."
                )
    else:
        # Not a local path → treat as Hugging Face dataset id (e.g. org/name)
        dataset = load_dataset(source, split=split)

    # --- pick split if we got a DatasetDict ---
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Split '{split}' not in dataset. Available: {list(dataset)}")
        dataset = dataset[split]

    # Math: drop rows with missing/empty \\boxed{...} gold before normalize.
    if task == "math":
        rows_before_filter = len(dataset)
        dataset = dataset.filter(_has_usable_reference)
        dropped_rows = rows_before_filter - len(dataset)
        if dropped_rows:
            print(f"Dropped {dropped_rows} rows with missing/empty boxed gold answers")

    # --- normalize rows, then fail fast on missing reward columns ---
    dataset = dataset.map(_normalize_row, fn_kwargs={"task": task})
    required = TASKS[task]["required_columns"]
    missing = [c for c in required if c not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Dataset missing required columns {missing} for task={task!r}. "
            f"Have {dataset.column_names}."
        )

    # AFM-Base path assumes plain-string prompts (no chat template).
    assert len(dataset) > 0, "Train dataset is empty"
    assert isinstance(dataset[0]["prompt"], str), (
        "prompt must be a plain string; message lists make TRL apply a chat "
        f"template (AFM-Base template is unclear). Got {type(dataset[0]['prompt'])}"
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
            except Exception:
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
# Env factory — metrics make_env_reward_fn expects load_problem(entry: dict)
# ---------------------------------------------------------------------------
def make_env_for_task(task: str):
    """Return an env instance keyed the same way as dataset column `task`."""
    if task == "countdown":
        return CountdownEnv()
    if task == "knights_knaves":
        return KnightsKnavesEnv()
    if task == "shortest_path":
        return ShortestPathEnv()
    if task == "math":
        # Friend will add load_problem(entry: dict) on CompetitionMathEnv.
        return CompetitionMathEnv()
    if task == "coding":
        raise NotImplementedError(
            "coding has no env yet; use the no-op reward path for --task coding"
        )
    raise ValueError(f"Unknown task: {task}")


def make_coding_reward():
    """Placeholder coding reward — scores nothing (all zeros) until MBPP is wired."""

    def reward_coding(completions, **kwargs):
        return [0.0] * len(completions)

    return reward_coding


# ---------------------------------------------------------------------------
# Model — base checkpoint + LoRA (fresh or continued from SFT adapter)
# ---------------------------------------------------------------------------
def build_lora_config() -> LoraConfig:
    """Broadly compatible causal-LM LoRA config (same idea as math/train.py)."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def load_model(init_adapter: str | None):
    # No device_map: let the Trainer / vLLM colocate placement own the device.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="bfloat16",
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
    args = parse_args()
    if args.task not in TASKS:
        raise ValueError(f"Unknown task: {args.task!r}")

    # --- sanity checks on fixed batch geometry ---
    if ROLLOUTS_PER_STEP % args.G != 0:
        raise ValueError(
            f"ROLLOUTS_PER_STEP ({ROLLOUTS_PER_STEP}) must be divisible by G ({args.G})"
        )
    if MICRO_BATCH_COMPLETIONS * GRAD_ACCUM_STEPS != ROLLOUTS_PER_STEP:
        raise ValueError("Microbatch × grad accum must equal ROLLOUTS_PER_STEP")

    # prompts per optimizer step changes with G; rollouts (64) do not
    prompts_per_step = ROLLOUTS_PER_STEP // args.G
    lr_tag = f"{args.lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    run_name = f"grpo-{args.task}-G{args.G}-lr{lr_tag}-s{args.seed}"
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
    dataset = load_train_dataset(args.dataset, args.dataset_split, task=args.task)
    print(dataset)
    print(f"columns={dataset.column_names} | rows={len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.init_adapter)

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
        max_prompt_length=512,
        max_completion_length=1024,  # room for <think>…<answer>
        optim="adamw_8bit",
        max_steps=args.max_steps,  # matched compute; not num_train_epochs
        warmup_ratio=0.03,  # short warmup; LR still decays linearly to 0 by default
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
        report_to=["wandb"],
        remove_unused_columns=False,  # keep task/entry for metrics reward_fn
        use_vllm=True,
        vllm_mode="colocate",  # vLLM shares the training GPU
        vllm_gpu_memory_utilization=0.3,  # leave headroom for the train model
        run_name=run_name,
    )

    # --- train ---
    # Loop inside TRL: sample G completions/prompt → score → group-normalize → update.
    # Map CLI "none" → None (full budget). Library stages are sft_then_grpo:*.
    budget_stage = None if args.stage == "none" else args.stage
    token_cb = CompletionTokenCallback(ROLLOUTS_PER_STEP, output_dir)
    budget_cb = ComputeBudgetCallback(
        output_dir,
        budget=args.budget,
        stage=budget_stage,
        lora=True,
    )
    metrics = GRPOMetrics(output_dir)
    if args.task == "coding":
        reward_fn = make_coding_reward()
    else:
        reward_fn = make_env_reward_fn({args.task: make_env_for_task(args.task)}, metrics)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[token_cb, budget_cb, GRPOMetricsCallback(metrics)],
    )

    if patch_qwen35_vllm_weight_sync(trainer, model):
        print("Enabled Qwen3.5 Hugging Face → vLLM weight-name compatibility mapping.")

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()