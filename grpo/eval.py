"""
Held-out evaluation for GRPO checkpoints.

Reuses dataset normalize / env / model-load helpers from grpo.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from grpo.countdown import CountdownEnv
from grpo.knights_knaves import KnightsKnavesEnv
from grpo.shortest_path import ShortestPathEnv
from grpo.math import CompetitionMathEnv, extract_boxed_answer
from grpo.metrics import FAIL_TYPES, _failure_type

# Keep in sync with grpo.py
MODEL_ID = "Qwen/Qwen3.5-4B-Base"
MAX_COMPLETION_TOKENS = 4096
CONTEXT_TOKENS = 5120  # prompt + completion must fit (matches grpo.py default)

QWEN35_MODEL_TYPES = ("qwen3_5", "qwen3_5_text")
QWEN35_VLLM_PREFIX_MAP = (
    ("lm_head.", "language_model.lm_head."),
    ("model.", "language_model.model."),
)

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
# Dataset helpers (copied from grpo.py)
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
        obs = env.load_problem(entry)
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


def load_examples(source: str, split: str, task: str) -> Dataset:
    """Load eval data from disk or the HF Hub, then normalize + validate schema."""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task!r}")

    path = Path(source)

    if path.exists():
        dataset = _load_local_dataset(path, split)
    else:
        dataset = load_dataset(source, split=split)

    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Split '{split}' not in dataset. Available: {list(dataset)}")
        dataset = dataset[split]

    if task == "math":
        rows_before_filter = len(dataset)
        dataset = dataset.filter(_has_usable_reference)
        dropped_rows = rows_before_filter - len(dataset)
        if dropped_rows:
            print(f"Dropped {dropped_rows} rows with missing/empty boxed gold answers")

    dataset = dataset.map(_normalize_row, fn_kwargs={"task": task})
    required = TASKS[task]["required_columns"]
    missing = [c for c in required if c not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Dataset missing required columns {missing} for task={task!r}. "
            f"Have {dataset.column_names}."
        )

    assert len(dataset) > 0, "Eval dataset is empty"
    assert isinstance(dataset[0]["prompt"], str), (
        "prompt must be a plain string; message lists make chat templates apply. "
        f"Got {type(dataset[0]['prompt'])}"
    )
    return dataset


def build_prompt(example: dict, task: str) -> str:
    """Reproduce the exact training-time prompt formatting via env normalize."""
    return _normalize_row(example, task)["prompt"]


# ---------------------------------------------------------------------------
# Env factory (copied from grpo.py)
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
        return CompetitionMathEnv()
    if task == "coding":
        raise NotImplementedError(
            "coding has no env yet; wire MBPP before verifying coding eval"
        )
    raise ValueError(f"Unknown task: {task}")


def verify_response(task: str, example: dict, completion: str) -> dict:
    """Run the existing task-specific verifier (env.load_problem + env.step)."""
    env = make_env_for_task(task)
    entry = example.get("entry")
    if entry is None:
        entry = _normalize_row(example, task)["entry"]
    env.load_problem(entry)
    return env.step(completion)


# ---------------------------------------------------------------------------
# Model load (adapted from grpo.py load_model — inference, not trainable LoRA)
# ---------------------------------------------------------------------------
def load_policy(base_model_id: str = MODEL_ID, checkpoint: str | None = None):
    """Load base weights; optionally attach a saved LoRA adapter (eval mode)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
    )
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Eval protocol
# ---------------------------------------------------------------------------
def problem_id_for_row(row: dict, task: str) -> str:
    """
    Stable ID for comparing the same question across checkpoints.

    Prefer an explicit dataset id; otherwise hash task-specific content
    (not row order, which can change with filters / max-examples).
    """
    for key in ("problem_id", "task_id", "id", "uid", "example_id"):
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)

    if task in RG_TASKS:
        payload = {
            "task": task,
            "question": row.get("question"),
            "answer": row.get("answer"),
            "metadata": row.get("metadata"),
        }
    elif task == "math":
        payload = {
            "task": task,
            "problem": row.get("problem"),
            "solution": row.get("solution"),
            "level": row.get("level"),
            "type": row.get("type"),
        }
    else:
        entry = row.get("entry")
        payload = {
            "task": task,
            "prompt": row.get("prompt"),
            "entry": entry if isinstance(entry, (dict, list, str, int, float)) else str(entry),
        }

    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return f"{task}:{digest}"


def _eos_token_ids(tokenizer) -> set[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return set(eos)


def build_eval_generation_settings(
    tokenizer,
    *,
    max_new_tokens: int,
    context_tokens: int,
) -> dict:
    """
    Explicit main-eval decode settings (do not inherit model.generation_config).

    Vs GRPO train: greedy / one completion / no custom stops; EOS enabled.
    """
    return {
        "max_new_tokens": int(max_new_tokens),
        "context_tokens": int(context_tokens),
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": None,  # unused when do_sample=False
        "top_p": None,
        "top_k": None,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "stop_sequences": None,
        "backend": "hf_generate",
    }


def _generation_config_from_settings(settings: dict) -> GenerationConfig:
    """Build a GenerationConfig so model defaults cannot override eval protocol."""
    return GenerationConfig(
        max_new_tokens=settings["max_new_tokens"],
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        eos_token_id=settings["eos_token_id"],
        pad_token_id=settings["pad_token_id"],
    )


def fit_completion_to_context(
    prompt_tokens: int,
    max_new_tokens: int,
    context_tokens: int,
) -> int:
    """
    Ensure prompt_tokens + completion budget fits the configured context window.

    Returns the effective max_new_tokens (may be clamped). Raises if the prompt
    alone leaves no room for any completion token.
    """
    if prompt_tokens >= context_tokens:
        raise ValueError(
            f"prompt_tokens={prompt_tokens} does not fit context_tokens={context_tokens}"
        )
    room = context_tokens - prompt_tokens
    if room < 1:
        raise ValueError(
            f"no room for completions: prompt_tokens={prompt_tokens}, "
            f"context_tokens={context_tokens}"
        )
    return min(int(max_new_tokens), room)


def generate_response(model, tokenizer, example, task, config) -> dict:
    """
    Generate one greedy completion under the explicit eval protocol.

    Returns a dict with text, token counts, EOS-vs-length termination, and the
    exact generation settings used for this call.
    """
    if not isinstance(config, dict):
        config = {
            "max_new_tokens": MAX_COMPLETION_TOKENS,
            "context_tokens": CONTEXT_TOKENS,
        }

    requested_max_new = int(config.get("max_new_tokens", MAX_COMPLETION_TOKENS))
    context_tokens = int(config.get("context_tokens", CONTEXT_TOKENS))

    prompt = build_prompt(example, task)
    device = next(model.parameters()).device
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    prompt_tokens = int(model_inputs.input_ids.shape[-1])

    effective_max_new = fit_completion_to_context(
        prompt_tokens, requested_max_new, context_tokens
    )
    settings = build_eval_generation_settings(
        tokenizer,
        max_new_tokens=effective_max_new,
        context_tokens=context_tokens,
    )
    settings["max_new_tokens_requested"] = requested_max_new
    settings["prompt_tokens"] = prompt_tokens
    settings["clamped_to_context"] = effective_max_new < requested_max_new

    generated = model.generate(
        **model_inputs,
        generation_config=_generation_config_from_settings(settings),
        return_dict_in_generate=True,
    )
    sequences = generated.sequences
    output_ids = sequences[0][prompt_tokens:].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    n_tokens = len(output_ids)
    ended_with_eos = bool(output_ids) and output_ids[-1] in _eos_token_ids(tokenizer)
    hit_length_limit = n_tokens >= effective_max_new
    if ended_with_eos:
        termination_reason = "eos"
    elif hit_length_limit:
        termination_reason = "length"
    else:
        termination_reason = "unknown"

    return {
        "response": response,
        "n_tokens": n_tokens,
        "prompt_tokens": prompt_tokens,
        "hit_length_limit": hit_length_limit,
        "truncated": termination_reason == "length",
        "termination_reason": termination_reason,
        "ended_with_eos": ended_with_eos,
        "generation_settings": settings,
    }


def evaluate_dataset(
    model,
    tokenizer,
    dataset,
    task,
    config,
    results_path: str | Path | None = None,
):
    """Run inference, grade each result; optionally append each record to JSONL."""
    results = []
    out_file = None
    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = results_path.open("w", encoding="utf-8")

    try:
        for i, row in enumerate(dataset):
            # Row may be a HF feature dict; normalize access.
            row_dict = dict(row) if not isinstance(row, dict) else row
            problem_id = problem_id_for_row(row_dict, task)
            record: dict = {
                "index": i,
                "problem_id": problem_id,
                "task": task,
                "prompt": None,
                "response": None,
                "prompt_tokens": None,
                "n_tokens": None,
                "hit_length_limit": False,
                "truncated": False,
                "ended_with_eos": False,
                "termination_reason": None,
                "generation_settings": None,
                "answer": None,
                "answer_tokens": 0,
                "reward": None,
                "acceptor": False,
                "check": None,
                "error": None,
                "error_stage": None,
            }
            try:
                try:
                    record["prompt"] = row_dict.get("prompt") or build_prompt(
                        row_dict, task
                    )
                except Exception:
                    record["error_stage"] = "prompt"
                    raise

                try:
                    gen = generate_response(model, tokenizer, row_dict, task, config)
                except Exception:
                    record["error_stage"] = "generate"
                    raise

                response = gen["response"]
                record["response"] = response
                record["prompt_tokens"] = gen["prompt_tokens"]
                record["n_tokens"] = gen["n_tokens"]
                record["hit_length_limit"] = gen["hit_length_limit"]
                record["truncated"] = gen["truncated"]
                record["ended_with_eos"] = gen["ended_with_eos"]
                record["termination_reason"] = gen["termination_reason"]
                record["generation_settings"] = gen["generation_settings"]

                try:
                    out = verify_response(task, row_dict, response)
                except Exception:
                    record["error_stage"] = "verify"
                    raise

                answer = out.get("answer") or ""
                record["answer"] = out.get("answer")
                record["answer_tokens"] = (
                    len(tokenizer.encode(answer)) if answer else 0
                )
                record["reward"] = out.get("reward")
                record["acceptor"] = bool(out.get("acceptor"))
                record["check"] = out.get("check")
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                if record["error_stage"] is None:
                    record["error_stage"] = "unknown"
                # Keep verifier-style fields so summarize/_failure_type see an unscored fail.
                record["check"] = {
                    "error": record["error"],
                    "error_stage": record["error_stage"],
                }
                print(
                    f"[eval] {problem_id} (index={i}) failed at "
                    f"{record['error_stage']}: {record['error']}",
                    file=sys.stderr,
                )


            results.append(record)
            if out_file is not None:
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()
                os.fsync(out_file.fileno())
    finally:
        if out_file is not None:
            out_file.close()

    return results



def _percentile(ordered: list[int | float], q: float) -> float:
    if not ordered:
        return 0.0
    return float(ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))])


def _length_histogram(token_lens: list[int], edges: tuple[int, ...] | None = None) -> dict[str, int]:
    """Count completions into half-open buckets [lo, hi)."""
    if edges is None:
        edges = (0, 128, 256, 512, 1024, 2048, 4096, 8192, 10**9)
    counts = {
        f"{edges[i]}-{edges[i + 1] if edges[i + 1] < 10**9 else 'inf'}": 0
        for i in range(len(edges) - 1)
    }
    keys = list(counts)
    for n_tok in token_lens:
        for i in range(len(edges) - 1):
            if edges[i] <= n_tok < edges[i + 1]:
                counts[keys[i]] += 1
                break
    return counts


def _output_length_stats(results: list[dict], n: int) -> dict:
    token_lens = [int(r["n_tokens"]) for r in results if r.get("n_tokens") is not None]
    if not token_lens:
        return {
            "n_with_tokens": 0,
            "mean": 0.0,
            "median": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0,
            "max": 0,
            "hit_length_limit_rate": 0.0,
            "truncation_rate": 0.0,
            "termination_reasons": {},
            "histogram": _length_histogram([]),
        }

    ordered = sorted(token_lens)
    termination_reasons: dict[str, int] = {}
    for r in results:
        reason = r.get("termination_reason")
        if reason:
            termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

    return {
        "n_with_tokens": len(token_lens),
        "mean": sum(token_lens) / len(token_lens),
        "median": _percentile(ordered, 0.5),
        "p50": _percentile(ordered, 0.5),
        "p90": _percentile(ordered, 0.9),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "min": int(ordered[0]),
        "max": int(ordered[-1]),
        "hit_length_limit_rate": sum(1 for r in results if r.get("hit_length_limit")) / n,
        "truncation_rate": sum(1 for r in results if r.get("truncated")) / n,
        "termination_reasons": termination_reasons,
        "histogram": _length_histogram(token_lens),
    }


def summarize_results(results: list[dict]) -> dict:
    """Calculate accuracy, failure rates, and output statistics from eval records."""
    n = len(results)
    empty = {
        "n": 0,
        "n_correct": 0,
        "n_errors": 0,
        "accuracy": 0.0,
        "error_rate": 0.0,
        "mean_reward": 0.0,
        "output": _output_length_stats([], 0),
        "error_stages": {},
        "fail_counts": {ft: 0 for ft in FAIL_TYPES},
        "failure_rates": {ft: 0.0 for ft in FAIL_TYPES},
    }
    if n == 0:
        return empty

    n_correct = sum(1 for r in results if r.get("acceptor"))
    n_errors = sum(1 for r in results if r.get("error"))
    error_stages: dict[str, int] = {}
    for r in results:
        stage = r.get("error_stage")
        if stage:
            error_stages[stage] = error_stages.get(stage, 0) + 1

    rewards = [float(r["reward"]) for r in results if r.get("reward") is not None]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

    fail_counts = {ft: 0 for ft in FAIL_TYPES}
    for r in results:
        fail = _failure_type(r)
        if fail is not None:
            fail_counts[fail] = fail_counts.get(fail, 0) + 1

    failure_rates = {ft: fail_counts.get(ft, 0) / n for ft in fail_counts}

    return {
        "n": n,
        "n_correct": n_correct,
        "n_errors": n_errors,
        "accuracy": n_correct / n,
        "error_rate": n_errors / n,
        "mean_reward": mean_reward,
        "output": _output_length_stats(results, n),
        "error_stages": error_stages,
        "fail_counts": fail_counts,
        "failure_rates": failure_rates,
    }






# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Held-out eval for GRPO checkpoints (Reasoning Gym, math, coding stub)."
        )
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=("countdown", "knights_knaves", "shortest_path", "math", "coding"),
        help="Task / env to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Eval data: local .parquet / .json / .jsonl file, a directory containing "
            "<split>.parquet|.jsonl|.json (see --dataset-split), "
            "or a Hugging Face dataset id."
        ),
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="eval",
        help="Split name when --dataset is a HF hub id or a directory (default: eval).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional LoRA adapter / checkpoint dir to attach (eval mode).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help=f"Base model id or path (default: {MODEL_ID}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        "--completion-tokens",
        type=int,
        default=MAX_COMPLETION_TOKENS,
        dest="max_new_tokens",
        help=(
            "Max new tokens to generate per completion "
            f"(default: {MAX_COMPLETION_TOKENS})."
        ),
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=CONTEXT_TOKENS,
        help=(
            "Configured model context window; prompt_tokens + max_new_tokens must fit "
            f"(default: {CONTEXT_TOKENS}). Completions are clamped per example if needed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write results.jsonl and summary.json (default: EVAL/<task>).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of eval examples (smoke runs).",
    )
    return parser.parse_args()


def build_run_metadata(
    args: argparse.Namespace,
    *,
    config: dict,
    num_examples: int,
    output_dir: Path,
) -> dict:
    """Exact model / adapter / eval config for comparing runs."""
    checkpoint = args.checkpoint
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "num_examples": num_examples,
        "max_examples": args.max_examples,
        "model": args.model,
        "checkpoint": checkpoint,
        "checkpoint_resolved": str(Path(checkpoint).resolve()) if checkpoint else None,
        "output_dir": str(output_dir.resolve()),
        "eval_config": {
            **config,
            # Main eval protocol (not GRPO train: temp=1.0, G>1).
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "stop_sequences": None,
            "backend": "hf_generate",
            "termination_reporting": "eos|length|unknown",
            "inherits_model_generation_config": False,
        },
        "defaults": {
            "MODEL_ID": MODEL_ID,
            "MAX_COMPLETION_TOKENS": MAX_COMPLETION_TOKENS,
            "CONTEXT_TOKENS": CONTEXT_TOKENS,
        },
        "cli": vars(args),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or f"EVAL/eval-{args.task}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_new_tokens >= args.context_tokens:
        raise ValueError(
            f"--max-new-tokens ({args.max_new_tokens}) must be < "
            f"--context-tokens ({args.context_tokens}) so prompts can fit"
        )

    config = {
        "max_new_tokens": args.max_new_tokens,
        "context_tokens": args.context_tokens,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "stop_sequences": None,
        "inherits_model_generation_config": False,
    }

    dataset = load_examples(args.dataset, args.dataset_split, args.task)
    if args.max_examples is not None:
        n = min(args.max_examples, len(dataset))
        dataset = dataset.select(range(n))
    print(
        f"eval task={args.task} | rows={len(dataset)} | "
        f"checkpoint={args.checkpoint!r} | "
        f"max_new_tokens={args.max_new_tokens} | context_tokens={args.context_tokens}"
    )

    model, tokenizer = load_policy(args.model, checkpoint=args.checkpoint)
    # Resolve concrete EOS/PAD ids into metadata after tokenizer load.
    config["eos_token_id"] = tokenizer.eos_token_id
    config["pad_token_id"] = tokenizer.pad_token_id
    config["generation_settings"] = build_eval_generation_settings(
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        context_tokens=args.context_tokens,
    )

    metadata = build_run_metadata(
        args,
        config=config,
        num_examples=len(dataset),
        output_dir=output_dir,
    )
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    results_path = output_dir / "results.jsonl"
    results = evaluate_dataset(
        model,
        tokenizer,
        dataset,
        args.task,
        config,
        results_path=results_path,
    )
    summary = summarize_results(results)
    summary["metadata"] = metadata

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {metadata_path}, {results_path}, and {summary_path}")


if __name__ == "__main__":
    main()


