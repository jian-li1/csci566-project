"""
GRPO training script (TRL) for multi-task runs (Reasoning Gym, math, KodCode, OpenCodeInstruct, OpenMathReasoning).

Ablation design:
  - Hold rollouts/optimizer-step fixed at 64 for every arm.
  - Vary only --G (completions per prompt): 2, 4, 8, or 16.
  - Stop by --max-steps or --epochs (Slurm-controllable).
  - One run = one --task.

Train vs held-out eval (see grpo/eval.py):
  - Train samples with temperature=1.0 and G>1 for group-relative advantages.
  - Eval uses greedy, one completion per problem — do not point this script
    at a held-out split if you only want scoring (it will train / update LoRA).

Data flow per step:
  dataset row → normalize {prompt, task, entry}
    → sample G completions → make_env_reward_fn (env.step)
    → GRPO group-normalize + LoRA update
    → callbacks: completion tokens, compute budget, metrics

Prefer:  python -m grpo.grpo --task ... --G ... --dataset ...
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root on path before local imports (python grpo/grpo.py from anywhere).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from compute_budget import ComputeBudgetCallback
from grpo.countdown import CountdownEnv
from grpo.knights_knaves import KnightsKnavesEnv
from grpo.kodcode import KodCodeEnv
from grpo.opencodeinstruct import OpenCodeInstructEnv
from grpo.openmathreasoning import OpenMathReasoningEnv
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
TEMPERATURE = 1.0  # stochastic rollouts for GRPO groups (eval uses greedy)

# Shared generation / vLLM context caps (CLI defaults; override per Slurm job).
# prompt + completion must fit vllm_max_model_length (= --context-tokens).
MAX_COMPLETION_TOKENS = 4096
VLLM_CONTEXT_TOKENS = 5120

# Qwen3.5 is loaded as a text-only causal LM by Transformers, while vLLM uses
# the checkpoint's multimodal wrapper and therefore expects different prefixes.
QWEN35_MODEL_TYPES = ("qwen3_5", "qwen3_5_text")
QWEN35_VLLM_PREFIX_MAP = (
    ("lm_head.", "language_model.lm_head."),
    ("model.", "language_model.model."),
)

# --task choices. After normalize, every row has prompt / task / entry for the
# reward fn (TRL passes extra dataset columns through as reward kwargs).
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
    "kodcode": {
        "required_columns": ("prompt", "task", "entry"),
    },
    "opencodeinstruct": {
        "required_columns": ("prompt", "task", "entry"),
    },
    "openmathreasoning": {
        "required_columns": ("prompt", "task", "entry"),
    },
}

# ---------------------------------------------------------------------------
# CLI — sweep knobs live here; fixed geometry stays in constants above
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TRL GRPO on Reasoning Gym (countdown, knights_knaves, shortest path), "
            "Math, KodCode, OpenCodeInstruct, and OpenMathReasoning"
        )
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=(
            "countdown",
            "knights_knaves",
            "shortest_path",
            "math",
            "kodcode",
            "opencodeinstruct",
            "openmathreasoning",
        ),
        help=(
            "Task to train on. Choices: countdown, knights_knaves, shortest_path, "
            "math, kodcode, opencodeinstruct, openmathreasoning"
        ),
    )
    parser.add_argument(
        "--G",
        type=int,
        required=True,
        choices=(2, 4, 8, 16),
        help=(
            "Group size (completions per prompt). Rollouts/step stay fixed at 64 "
            "(32×2, 16×4, 8×8, or 4×16)."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Train data: local .parquet / .json / .jsonl file, a directory containing "
            "<split>.parquet|.jsonl|.json (see --dataset-split), "
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
        choices=("none", "sft_then_grpo:grpo"),
        help=(
            "Budget stage for this GRPO script: none = full budget (pure GRPO); "
            "sft_then_grpo:grpo = 75%% of B (GRPO half of SFT→GRPO). "
            "Run SFT separately with stage sft_then_grpo:sft."
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
    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help=(
            "If set, train for this many epochs (max_steps=-1). "
            "Omit to use --max-steps instead."
        ),
    )
    parser.add_argument(
        "--completion-tokens",
        type=int,
        default=MAX_COMPLETION_TOKENS,
        help=f"Max generated completion tokens (default: {MAX_COMPLETION_TOKENS}).",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=VLLM_CONTEXT_TOKENS,
        help=f"vLLM max model length / context (default: {VLLM_CONTEXT_TOKENS}).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--init-adapter",
        type=str,
        default=None,
        help="Path to an SFT LoRA adapter to continue from (SFT→GRPO arm).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Resume after a GPU timeout / preemption from a Trainer checkpoint. "
            "Pass a checkpoint-* dir, or use the flag alone (or 'auto') to pick "
            "the latest checkpoint under --output-dir. Re-submit with the same "
            "CLI args as the original run."
        ),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help=(
            "Disable colocated vLLM (use HF generate). Recommended for Windows / "
            "first laptop GPU smokes; CARC full runs usually keep vLLM on."
        ),
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="wandb",
        choices=("wandb", "none"),
        help="Logging backend (default: wandb). Use none for offline laptop smokes.",
    )
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


def _has_usable_expected_answer(row: dict) -> bool:
    """Reject OpenMathReasoning rows with a missing/empty expected answer."""
    answer = row.get("expected_answer")
    return isinstance(answer, str) and bool(answer.strip())


def _normalize_row(row: dict, task: str) -> dict:
    """
    Map one raw row → {prompt, task, entry, ...}.

    Builds `prompt` via the task env; `entry` is what metrics/reward pass to
    load_problem (RG / math / coding / OpenMathReasoning payload).
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
        obs = env.load_problem(entry)
        out["entry"] = entry
        out["prompt"] = obs["prompt"]
    elif task == "kodcode":
        env = KodCodeEnv()
        entry = {
            "question": out["question"],
            "test": out["test"],
        }
        if out.get("question_id") is not None:
            entry["question_id"] = out["question_id"]
        obs = env.load_problem(entry)
        out["entry"] = entry
        out["prompt"] = obs["prompt"]
    elif task == "opencodeinstruct":
        env = OpenCodeInstructEnv()
        entry = {
            "input": out["input"],
            "unit_tests": out["unit_tests"],
        }
        if out.get("id") is not None:
            entry["id"] = out["id"]
        obs = env.load_problem(entry)
        out["entry"] = entry
        out["prompt"] = obs["prompt"]
    elif task == "openmathreasoning":
        env = OpenMathReasoningEnv()
        entry = {
            "problem": out["problem"],
            "expected_answer": out["expected_answer"],
        }
        if out.get("problem_source") is not None:
            entry["problem_source"] = out["problem_source"]
        if out.get("problem_type") is not None:
            entry["problem_type"] = out["problem_type"]
        obs = env.load_problem(entry)
        out["entry"] = entry
        out["prompt"] = obs["prompt"]
    else:
        raise ValueError(f"Unknown task: {task}")

    return out


def _load_local_dataset(path: Path, split: str) -> Dataset:
    """Load one local file, or ``<directory>/<split>.{parquet,jsonl,json}``."""
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
    if suffix in {".jsonl", ".json"}:
        return load_dataset("json", data_files=str(path), split="train")
    raise ValueError(
        f"Unsupported local file type '{suffix}'. Use .parquet, .json, or .jsonl."
    )


def load_train_dataset(source: str, split: str, task: str) -> Dataset:
    """
    Load user data from disk or the HF Hub, then normalize + validate schema.

    Raw columns depend on task (RG: question/answer/metadata; math: problem/solution;
    kodcode: question/test; opencodeinstruct: input/unit_tests;
    openmathreasoning: problem/expected_answer).
    After normalize every row has prompt (plain str), task, and entry for rewards.
    """
    path = Path(source)

    # --- resolve source ---
    if path.exists():
        dataset = _load_local_dataset(path, split)
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

    # OpenMathReasoning: drop rows with missing/empty expected_answer.
    if task == "openmathreasoning":
        rows_before_filter = len(dataset)
        dataset = dataset.filter(_has_usable_expected_answer)
        dropped_rows = rows_before_filter - len(dataset)
        if dropped_rows:
            print(f"Dropped {dropped_rows} rows with missing/empty expected answers")

    # --- normalize rows, then fail fast on missing reward columns ---
    dataset = dataset.map(_normalize_row, fn_kwargs={"task": task})
    required = TASKS[task]["required_columns"]
    missing = [c for c in required if c not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Dataset missing required columns {missing} for task={task!r}. "
            f"Have {dataset.column_names}."
        )

    # Plain-string prompts: message lists make TRL apply a chat template.
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
    """
    Return an env instance keyed the same way as dataset column `task`.

    Envs own prompting + verification: load_problem(entry) then step(completion)
    → reward / acceptor / check (see metrics.make_env_reward_fn).
    """
    if task == "countdown":
        return CountdownEnv()
    if task == "knights_knaves":
        return KnightsKnavesEnv()
    if task == "shortest_path":
        return ShortestPathEnv()
    if task == "math":
        return CompetitionMathEnv()
    if task == "kodcode":
        return KodCodeEnv()
    if task == "opencodeinstruct":
        return OpenCodeInstructEnv()
    if task == "openmathreasoning":
        return OpenMathReasoningEnv()
    raise ValueError(f"Unknown task: {task}")


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
    """
    Load MODEL_ID + LoRA. Device placement is left to Trainer / colocated vLLM
    (no device_map here). For inference-only loading see grpo/eval.py load_policy.
    """
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


def resolve_resume_checkpoint(
    output_dir: str | Path, resume: str | None
) -> str | None:
    """
    Resolve --resume-from-checkpoint to a concrete checkpoint path.

    ``auto`` / bare flag → latest ``checkpoint-*`` under ``output_dir``.
    Explicit path → that directory. ``None`` → start fresh.
    """
    if resume is None:
        return None

    from transformers.trainer_utils import get_last_checkpoint

    if resume.lower() in {"auto", "true", "1", "yes"}:
        last = get_last_checkpoint(str(output_dir))
        if last is None:
            print(
                f"No Trainer checkpoint found under {output_dir}; "
                "starting a fresh run."
            )
            return None
        print(f"Resuming from latest checkpoint: {last}")
        return last

    path = Path(resume)
    if not path.is_dir():
        raise ValueError(
            f"--resume-from-checkpoint path does not exist or is not a directory: "
            f"{resume}"
        )
    print(f"Resuming from checkpoint: {path}")
    return str(path)


def patch_qwen35_vllm_weight_sync(trainer: GRPOTrainer, model: PeftModel) -> bool:
    """
    Map text-only Qwen3.5 weight names to vLLM's multimodal wrapper.

    Without this, colocated vLLM weight sync can silently miss parameters when
    Transformers loads qwen3_5_text but vLLM expects language_model.* prefixes.
    """
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
    """CLI entry: load train data + LoRA → GRPOTrainer.train() → save adapter."""
    args = parse_args()
    if args.task not in TASKS:
        raise ValueError(f"Unknown task: {args.task!r}")

    # --- sanity checks on fixed batch geometry ---
    # G only changes how 64 rollouts are grouped (prompts/step = 64/G), not the
    # number of completions per optimizer step (must stay matched across arms).
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

    resume_checkpoint = resolve_resume_checkpoint(
        output_dir, args.resume_from_checkpoint
    )

    stop = (
        f"epochs={args.epochs}"
        if args.epochs is not None
        else f"max_steps={args.max_steps}"
    )
    print(
        f"{run_name} | rollouts/step={ROLLOUTS_PER_STEP} | "
        f"prompts/step={prompts_per_step} | {stop} | "
        f"completion_tokens={args.completion_tokens} | "
        f"context_tokens={args.context_tokens} | "
        f"init_adapter={args.init_adapter!r} | "
        f"resume_from_checkpoint={resume_checkpoint!r}"
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
        max_completion_length=args.completion_tokens,  # room for <think>…<answer>
        # bitsandbytes 8-bit Adam is awkward on Windows; use torch Adam without vLLM.
        optim="adamw_torch" if args.no_vllm else "adamw_8bit",
        # --epochs N → epoch schedule; otherwise fixed --max-steps (default 400).
        max_steps=-1 if args.epochs is not None else args.max_steps,
        num_train_epochs=args.epochs if args.epochs is not None else 3.0,
        warmup_steps=0.03,  # short warmup; LR still decays linearly to 0 by default
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
        remove_unused_columns=False,  # keep task/entry for metrics reward_fn
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",  # vLLM shares the training GPU
        vllm_gpu_memory_utilization=0.3,  # leave headroom for the train model
        vllm_max_model_length=args.context_tokens,  # cap context so colocated KV cache fits
        run_name=run_name,
    )

    # --- train ---
    # Loop inside TRL: sample G completions/prompt → score → group-normalize → update.
    # Map CLI "none" → None (full budget). Library stages are sft_then_grpo:*.
    budget_stage = None if args.stage == "none" else args.stage
    # Approx completion-token spend for accuracy-vs-tokens plots (completion_tokens.json).
    token_cb = CompletionTokenCallback(ROLLOUTS_PER_STEP, output_dir)
    # FLOP-weighted token budget; may stop training when --budget is set (compute.json).
    budget_cb = ComputeBudgetCallback(
        output_dir,
        budget=args.budget,
        stage=budget_stage,
        lora=True,
    )
    # Per-rollout logging → rollouts.jsonl + Trainer/W&B scalars via callback.
    metrics = GRPOMetrics(output_dir, max_completion_length=training_args.max_completion_length)
    # Single reward path for RG/math/coding: env.step + metrics.record per completion.
    reward_fn = make_env_reward_fn({args.task: make_env_for_task(args.task)}, metrics)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[token_cb, budget_cb, GRPOMetricsCallback(metrics)],
    )

    if not args.no_vllm and patch_qwen35_vllm_weight_sync(trainer, model):
        print("Enabled Qwen3.5 Hugging Face → vLLM weight-name compatibility mapping.")

    # Restores model / optimizer / scheduler / RNG from checkpoint-* when set
    # (GPU timeout / Slurm requeue). Keep the same CLI as the interrupted run.
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    # Final adapter weights (also mid-run saves every save_steps).
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
