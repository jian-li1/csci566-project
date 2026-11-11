"""
Held-out evaluation for GRPO checkpoints.

Reuses dataset normalize / env / model-load helpers from grpo.py, but does **not**
run TRL GRPOTrainer. This script is inference + env verification only.

Eval protocol (intentionally different from training):
  - Greedy decode (do_sample=False), one completion per problem
  - Max new tokens default 4096; context window default 5120
  - Explicit GenerationConfig (does not inherit model.generation_config)
  - Record EOS vs length-limit termination per completion
  - Frozen comparison EOS source: COMPARISON_EOS_SOURCE ("model") for Base/SFT/GRPO
  - Dataset fingerprint includes prompt + grading_reference (not prompt alone)

Artifacts under --output-dir (unique per run by default):
  metadata.json   — model, adapter, dataset, decode + EOS settings
  results.jsonl   — one record per example (flushed incrementally)
  summary.json    — model accuracy (scored only), infra errors, length stats

Incomplete runs: infrastructure failures (prompt/generate/verify exceptions) are
kept on disk but excluded from model accuracy; summary.complete is False.

Context clamping: long prompts may get a shorter completion budget so
prompt+completion ≤ --context-tokens. Surveyed before eval; optional
--exclude-clamped drops those rows for a uniform max_new_tokens.

Prefer:  python -m grpo.eval --task ... --dataset ... [--checkpoint ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

# Env severities that mean "unscored / fatal" — not a normal wrong answer.
UNSCORED_SEVERITIES = frozenset({"end_rollout", "stop_training"})

# Repo root on path so `grpo.*` imports work when launched as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset, DatasetDict, load_dataset

from grpo.countdown import CountdownEnv
from grpo.knights_knaves import KnightsKnavesEnv
from grpo.kodcode import KodCodeEnv
from grpo.opencodeinstruct import OpenCodeInstructEnv
from grpo.openmathreasoning import OpenMathReasoningEnv
from grpo.shortest_path import ShortestPathEnv
from grpo.math import CompetitionMathEnv, extract_boxed_answer
from grpo.metrics import FAIL_TYPES, _failure_type

# torch / transformers / peft are imported lazily in load_policy / generate so
# contract unit tests can import this module without a GPU stack.

# ---------------------------------------------------------------------------
# Constants — keep MODEL_ID / token caps in sync with grpo.py
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3.5-4B-Base"
MAX_COMPLETION_TOKENS = 4096
# prompt_tokens + max_new_tokens must fit; same default as grpo.py VLLM_CONTEXT_TOKENS
CONTEXT_TOKENS = 5120

# Frozen Base/SFT/GRPO comparison decode protocol (do not retune per checkpoint).
# --eos-source may override for debugging only; comparisons must keep this value.
COMPARISON_EOS_SOURCE = "model"

# Unused here (train-only vLLM sync), kept so copies stay aligned with grpo.py.
QWEN35_MODEL_TYPES = ("qwen3_5", "qwen3_5_text")
QWEN35_VLLM_PREFIX_MAP = (
    ("lm_head.", "language_model.lm_head."),
    ("model.", "language_model.model."),
)

# Task names match --task / dataset column `task` / make_env_for_task keys.
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


def load_examples(source: str, split: str, task: str) -> tuple[Dataset, dict]:
    """
    Load eval data from disk or the HF Hub, then normalize + validate schema.

    Returns (dataset, load_stats). Math eligibility (usable \\boxed{} gold) is applied
    with the same predefined rule for every checkpoint; stats record raw → filtered sizes.
    """
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

    n_raw = len(dataset)
    n_dropped_invalid_reference = 0
    filter_rules: list[str] = []

    if task == "math":
        # Predefined eligibility: must have a non-empty boxed gold answer.
        filter_rules.append("math: drop rows without usable \\boxed{} gold in solution")
        rows_before_filter = len(dataset)
        dataset = dataset.filter(_has_usable_reference)
        n_dropped_invalid_reference = rows_before_filter - len(dataset)
        if n_dropped_invalid_reference:
            print(
                f"Dropped {n_dropped_invalid_reference} rows with "
                "missing/empty boxed gold answers"
            )

    if task == "openmathreasoning":
        filter_rules.append(
            "openmathreasoning: drop rows without usable expected_answer"
        )
        rows_before_filter = len(dataset)
        dataset = dataset.filter(_has_usable_expected_answer)
        n_dropped_invalid_reference = rows_before_filter - len(dataset)
        if n_dropped_invalid_reference:
            print(
                f"Dropped {n_dropped_invalid_reference} rows with "
                "missing/empty expected answers"
            )

    dataset = dataset.map(_normalize_row, fn_kwargs={"task": task})
    required = TASKS[task]["required_columns"]
    missing = [c for c in required if c not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Dataset missing required columns {missing} for task={task!r}. "
            f"Have {dataset.column_names}."
        )

    if len(dataset) == 0:
        raise ValueError("Eval dataset is empty after loading/eligibility filters")
    assert isinstance(dataset[0]["prompt"], str), (
        "prompt must be a plain string; message lists make chat templates apply. "
        f"Got {type(dataset[0]['prompt'])}"
    )

    load_stats = {
        "n_raw": n_raw,
        "n_after_eligibility_filters": len(dataset),
        "n_dropped_invalid_reference": n_dropped_invalid_reference,
        "filter_rules": filter_rules,
    }
    return dataset, load_stats


def grading_reference_for_row(row: dict, task: str) -> dict:
    """Stable verifier-facing ground truth used in dataset fingerprints."""
    entry = row.get("entry")
    if entry is None:
        entry = _normalize_row(row, task)["entry"]
    if not isinstance(entry, dict):
        return {"entry": entry}
    if task in RG_TASKS:
        return {
            "question": entry.get("question"),
            "answer": entry.get("answer"),
            "metadata": entry.get("metadata"),
        }
    if task == "math":
        return {
            "problem": entry.get("problem"),
            "solution": entry.get("solution"),
            "level": entry.get("level"),
            "type": entry.get("type"),
        }
    if task == "kodcode":
        return {
            "question_id": entry.get("question_id"),
            "question": entry.get("question"),
            "test": entry.get("test"),
        }
    if task == "opencodeinstruct":
        return {
            "id": entry.get("id"),
            "input": entry.get("input"),
            "unit_tests": entry.get("unit_tests"),
        }
    if task == "openmathreasoning":
        return {
            "problem": entry.get("problem"),
            "expected_answer": entry.get("expected_answer"),
            "problem_source": entry.get("problem_source"),
            "problem_type": entry.get("problem_type"),
        }
    return dict(entry)


def build_prompt(example: dict, task: str) -> str:
    """Reproduce the exact training-time prompt formatting via env normalize."""
    return _normalize_row(example, task)["prompt"]


def canonical_prompt(example: dict, task: str) -> str:
    """
    Prompt text that must match the dataset fingerprint and model input.

    Prefer the already-normalized ``prompt`` column from load_examples. If a
    rebuild via build_prompt differs, raise — do not silently send a different
    string than the one fingerprinted in the manifest.
    """
    stored = example.get("prompt")
    rebuilt = build_prompt(example, task)
    if stored is None or stored == "":
        return rebuilt
    if stored != rebuilt:
        raise ValueError(
            "Prompt mismatch: row['prompt'] != build_prompt(row). "
            "Fingerprint and generation would disagree. "
            f"stored_len={len(stored)} rebuilt_len={len(rebuilt)}"
        )
    return stored


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
    if task == "kodcode":
        return KodCodeEnv()
    if task == "opencodeinstruct":
        return OpenCodeInstructEnv()
    if task == "openmathreasoning":
        return OpenMathReasoningEnv()
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
    """
    Load base weights; optionally attach a saved LoRA adapter (eval mode).

    Unlike grpo.load_model, adapters are not trainable and we place the model
    on CUDA/CPU ourselves (no Trainer / vLLM device map).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
    )
    if checkpoint:
        # GRPO/SFT output dir containing adapter_config.json + weights.
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Eval protocol — greedy generate → env.step → persistent records
# ---------------------------------------------------------------------------
def problem_id_for_row(row: dict, task: str) -> str:
    """
    Stable ID for comparing the same question across checkpoints.

    Prefer an explicit dataset id; otherwise hash task-specific content
    (not row order, which can change with filters / max-examples).
    """
    for key in ("problem_id", "task_id", "question_id", "id", "uid", "example_id"):
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    entry = row.get("entry")
    if isinstance(entry, dict):
        for key in ("question_id", "id"):
            value = entry.get(key)
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
    elif task == "kodcode":
        payload = {
            "task": task,
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "test": row.get("test"),
        }
    elif task == "opencodeinstruct":
        payload = {
            "task": task,
            "id": row.get("id"),
            "input": row.get("input"),
            "unit_tests": row.get("unit_tests"),
        }
    elif task == "openmathreasoning":
        payload = {
            "task": task,
            "problem": row.get("problem"),
            "expected_answer": row.get("expected_answer"),
            "problem_source": row.get("problem_source"),
            "problem_type": row.get("problem_type"),
        }
    else:
        payload = {
            "task": task,
            "prompt": row.get("prompt"),
            "entry": entry if isinstance(entry, (dict, list, str, int, float)) else str(entry),
        }

    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return f"{task}:{digest}"


def _as_id_set(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    try:
        return {int(v) for v in value}
    except TypeError:
        return set()


def resolve_eos_config(model, tokenizer, *, source: str = "model") -> dict:
    """
    Resolve effective EOS / PAD ids used for stopping and termination checks.

    source:
      - "model": use model.generation_config.eos_token_id if set, else tokenizer
      - "tokenizer": tokenizer.eos_token_id only
      - "union": tokenizer ∪ model (can stop earlier if extra ids are added)

    Do not assume union is always correct for Qwen — validate with natural EOS,
    length-limit, and EOS-at-cap smokes. Warn when tokenizer and model disagree.
    """
    tok_eos = _as_id_set(tokenizer.eos_token_id)
    gen_cfg = getattr(model, "generation_config", None)
    model_eos_raw = getattr(gen_cfg, "eos_token_id", None) if gen_cfg else None
    model_eos = _as_id_set(model_eos_raw)

    if source == "tokenizer":
        eos_ids = sorted(tok_eos)
        chosen = "tokenizer"
    elif source == "union":
        eos_ids = sorted(tok_eos | model_eos)
        chosen = "union"
    else:  # model (default)
        if model_eos:
            eos_ids = sorted(model_eos)
            chosen = "model.generation_config"
        else:
            eos_ids = sorted(tok_eos)
            chosen = "tokenizer_fallback"

    mismatch = bool(tok_eos and model_eos and tok_eos != model_eos)
    if mismatch:
        print(
            "[eval] WARNING: tokenizer EOS "
            f"{sorted(tok_eos)} != model.generation_config EOS {sorted(model_eos)}; "
            f"using source={source!r} → effective={eos_ids}. "
            "Confirm this matches intended Qwen stop behavior.",
            file=sys.stderr,
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None and eos_ids:
        pad_id = eos_ids[0]
    return {
        "eos_source": source,
        "eos_source_resolved": chosen,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "model_generation_config_eos_token_id": model_eos_raw,
        "tokenizer_model_eos_mismatch": mismatch,
        "eos_token_ids": eos_ids,
        "pad_token_id": pad_id,
        "pad_equals_eos": pad_id in eos_ids if pad_id is not None and eos_ids else False,
    }


def is_unresolved_verifier_result(out: dict) -> bool:
    """
    True when env.step reported an unscored / fatal outcome without raising.

    Normal wrong answers use reward=0 (or >0 partial) with acceptor=False —
    those stay scored. reward=None or severity end_rollout/stop_training means
    the example was not validly graded.
    """
    if out.get("reward") is None:
        return True
    severity = out.get("severity")
    return severity in UNSCORED_SEVERITIES


def _eos_token_ids_from_config(config: dict) -> set[int]:
    if "eos_token_ids" in config and config["eos_token_ids"] is not None:
        return _as_id_set(config["eos_token_ids"])
    return _as_id_set(config.get("eos_token_id"))


def build_eval_generation_settings(
    tokenizer,
    *,
    max_new_tokens: int,
    context_tokens: int,
    eos_config: dict | None = None,
) -> dict:
    """
    Explicit main-eval decode settings (do not inherit model.generation_config).

    Vs GRPO train: greedy / one completion / no custom stops; EOS enabled.
    """
    if eos_config is None:
        eos_ids = sorted(_as_id_set(tokenizer.eos_token_id))
        pad_id = tokenizer.pad_token_id
        if pad_id is None and eos_ids:
            pad_id = eos_ids[0]
    else:
        eos_ids = list(eos_config["eos_token_ids"])
        pad_id = eos_config["pad_token_id"]

    # GenerationConfig accepts int or list for eos_token_id.
    eos_for_generate: int | list[int] | None
    if not eos_ids:
        eos_for_generate = None
    elif len(eos_ids) == 1:
        eos_for_generate = eos_ids[0]
    else:
        eos_for_generate = eos_ids

    return {
        "max_new_tokens": int(max_new_tokens),
        "context_tokens": int(context_tokens),
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": None,  # unused when do_sample=False
        "top_p": None,
        "top_k": None,
        "eos_token_id": eos_for_generate,
        "eos_token_ids": eos_ids,
        "pad_token_id": pad_id,
        "stop_sequences": None,
        "backend": "hf_generate",
    }


def _generation_config_from_settings(settings: dict):
    """Build a GenerationConfig so model defaults cannot override eval protocol."""
    from transformers import GenerationConfig

    return GenerationConfig(
        max_new_tokens=settings["max_new_tokens"],
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        eos_token_id=settings["eos_token_id"],
        pad_token_id=settings["pad_token_id"],
    )


def survey_context_clamping(
    dataset: Dataset,
    task: str,
    tokenizer,
    max_new_tokens: int,
    context_tokens: int,
) -> dict:
    """
    Count how many prompts would get a clamped completion budget.

    Different problems can receive different max_new_tokens after clamping;
    the same problem_id still gets the same limit across checkpoints. Use
    --exclude-clamped for a uniform completion budget.
    """
    n_fit = 0
    n_clamped = 0
    n_overflow = 0  # prompt alone does not fit
    clamped_ids: list[str] = []
    overflow_ids: list[str] = []

    for i, row in enumerate(dataset):
        row_dict = dict(row) if not isinstance(row, dict) else row
        pid = problem_id_for_row(row_dict, task)
        prompt = canonical_prompt(row_dict, task)
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        if prompt_tokens >= context_tokens:
            n_overflow += 1
            overflow_ids.append(pid)
            continue
        effective = fit_completion_to_context(
            prompt_tokens, max_new_tokens, context_tokens
        )
        if effective < max_new_tokens:
            n_clamped += 1
            clamped_ids.append(pid)
        else:
            n_fit += 1

    return {
        "max_new_tokens_requested": max_new_tokens,
        "context_tokens": context_tokens,
        "n_fit_full_budget": n_fit,
        "n_clamped": n_clamped,
        "n_prompt_overflow": n_overflow,
        "clamp_rate": n_clamped / max(1, n_fit + n_clamped + n_overflow),
        "clamped_problem_ids": clamped_ids,
        "overflow_problem_ids": overflow_ids,
        "note": (
            "Clamping shortens max_new_tokens so prompt_tokens + completion "
            "≤ context_tokens. Same problem_id → same limit across checkpoints."
        ),
    }


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

    # Exact fingerprinted prompt (assert rebuild matches if both present).
    prompt = canonical_prompt(example, task)
    device = next(model.parameters()).device
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    prompt_tokens = int(model_inputs.input_ids.shape[-1])

    # Clamp completion budget so prompt + completion ≤ context_tokens.
    effective_max_new = fit_completion_to_context(
        prompt_tokens, requested_max_new, context_tokens
    )
    eos_config = config.get("eos_config")
    settings = build_eval_generation_settings(
        tokenizer,
        max_new_tokens=effective_max_new,
        context_tokens=context_tokens,
        eos_config=eos_config,
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
    # Drop prompt prefix — only the newly generated ids are the completion.
    output_ids = sequences[0][prompt_tokens:].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    n_tokens = len(output_ids)
    eos_ids = _eos_token_ids_from_config(settings)
    ended_with_eos = bool(output_ids) and output_ids[-1] in eos_ids
    hit_length_limit = n_tokens >= effective_max_new
    # Prefer EOS when both could apply (model stopped on EOS at the cap).
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
    """
    Run inference and grade each result.

    On failure (prompt / generate / verify), still append a record with
    error + error_stage so questions are never silently skipped. When
    results_path is set, each record is flushed+fsync'd before the next
    example so mid-run crashes keep completed rows.
    """
    results = []
    out_file = None
    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = results_path.open("w", encoding="utf-8")

    try:
        for i, row in enumerate(dataset):
            # HF Dataset rows may be dict-like; normalize to a plain dict.
            row_dict = dict(row) if not isinstance(row, dict) else row
            problem_id = problem_id_for_row(row_dict, task)
            # Pre-fill so partial failures still have a stable schema.
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
                "clamped_to_context": False,
                "answer": None,
                "answer_tokens": 0,
                "reward": None,
                "acceptor": False,
                "severity": None,
                "check": None,
                "error": None,
                "error_stage": None,
                # Infrastructure / unresolved verifier: not counted as wrong answers.
                "is_infra_error": False,
                "scored": False,
            }
            try:
                try:
                    # Same string that will be fingerprinted and fed to the model.
                    record["prompt"] = canonical_prompt(row_dict, task)
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
                record["clamped_to_context"] = bool(
                    (gen.get("generation_settings") or {}).get("clamped_to_context")
                )

                try:
                    out = verify_response(task, row_dict, response)
                except Exception:
                    record["error_stage"] = "verify"
                    raise

                record["answer"] = out.get("answer")
                # answer may be None / non-str on some paths; only count when str.
                answer = out.get("answer")
                if isinstance(answer, str) and answer:
                    record["answer_tokens"] = len(tokenizer.encode(answer))
                else:
                    record["answer_tokens"] = 0
                record["reward"] = out.get("reward")
                record["acceptor"] = bool(out.get("acceptor"))
                record["severity"] = out.get("severity")
                record["check"] = out.get("check")

                # Env can return unscored/fatal without raising (reward=None, etc.).
                if is_unresolved_verifier_result(out):
                    record["error_stage"] = "verify_unscored"
                    record["is_infra_error"] = True
                    record["scored"] = False
                    record["error"] = (
                        f"unresolved verifier result: reward={out.get('reward')!r} "
                        f"severity={out.get('severity')!r} "
                        f"check_error={(out.get('check') or {}).get('error')!r}"
                    )
                    print(
                        f"[eval] {problem_id} (index={i}) UNRESOLVED verifier: "
                        f"{record['error']}",
                        file=sys.stderr,
                    )
                else:
                    record["scored"] = True
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                if record["error_stage"] is None:
                    record["error_stage"] = "unknown"
                record["is_infra_error"] = True
                record["scored"] = False
                record["acceptor"] = False
                # Do NOT shape this like an env verifier_crash — it's incomplete eval.
                record["check"] = {
                    "infra_error": record["error"],
                    "error_stage": record["error_stage"],
                }
                print(
                    f"[eval] {problem_id} (index={i}) INFRA failed at "
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
    """Length stats over scored generations only (n = denom for rates)."""
    token_lens = [int(r["n_tokens"]) for r in results if r.get("n_tokens") is not None]
    empty_hist = _length_histogram([])
    if not token_lens or n <= 0:
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
            "n_clamped_to_context": 0,
            "clamp_rate": 0.0,
            "termination_reasons": {},
            "histogram": empty_hist,
        }

    ordered = sorted(token_lens)
    termination_reasons: dict[str, int] = {}
    for r in results:
        reason = r.get("termination_reason")
        if reason:
            termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

    n_clamped = sum(1 for r in results if r.get("clamped_to_context"))
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
        "n_clamped_to_context": n_clamped,
        "clamp_rate": n_clamped / n,
        "termination_reasons": termination_reasons,
        "histogram": _length_histogram(token_lens),
    }


def summarize_results(results: list[dict]) -> dict:
    """
    Aggregate per-example records into headline metrics.

    Model accuracy uses only scored examples (successful generate+verify).
    Infrastructure errors are reported separately; complete=False if any remain.
    Env failure buckets (_failure_type) apply only to scored rows.
    """
    n = len(results)
    empty = {
        "n": 0,
        "n_scored": 0,
        "n_correct": 0,
        "n_infra_errors": 0,
        "complete": False,  # empty is never a successful benchmark
        "accuracy": None,
        "accuracy_note": "accuracy = n_correct / n_scored (infra/unresolved excluded)",
        "infra_error_rate": 0.0,
        "mean_reward": 0.0,
        "output": _output_length_stats([], 0),
        "error_stages": {},
        "fail_counts": {ft: 0 for ft in FAIL_TYPES},
        "failure_rates": {ft: 0.0 for ft in FAIL_TYPES},
    }
    if n == 0:
        return empty

    scored = [r for r in results if r.get("scored")]
    infra = [r for r in results if r.get("is_infra_error") or r.get("error")]
    n_scored = len(scored)
    n_infra = len(infra)
    n_correct = sum(1 for r in scored if r.get("acceptor"))

    error_stages: dict[str, int] = {}
    for r in infra:
        stage = r.get("error_stage") or "unknown"
        error_stages[stage] = error_stages.get(stage, 0) + 1

    rewards = [float(r["reward"]) for r in scored if r.get("reward") is not None]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

    # Model failure taxonomy — only scored rows (never unresolved verifier states).
    fail_counts = {ft: 0 for ft in FAIL_TYPES}
    for r in scored:
        if r.get("acceptor"):
            continue
        fail = _failure_type(r)
        if fail is not None:
            fail_counts[fail] = fail_counts.get(fail, 0) + 1

    failure_rates = {
        ft: (fail_counts.get(ft, 0) / n_scored if n_scored else 0.0) for ft in fail_counts
    }

    return {
        "n": n,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "n_infra_errors": n_infra,
        # Nonempty, fully scored, no infra/unresolved verifier failures.
        "complete": n > 0 and n_infra == 0 and n_scored == n,
        "accuracy": (n_correct / n_scored) if n_scored else None,
        "accuracy_note": "accuracy = n_correct / n_scored (infra/unresolved excluded)",
        "infra_error_rate": n_infra / n,
        "mean_reward": mean_reward,
        "output": _output_length_stats(scored, max(n_scored, 1) if scored else 0),
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
            "Held-out eval for GRPO checkpoints (Reasoning Gym, math, "
            "KodCode, OpenCodeInstruct, OpenMathReasoning)."
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
            "Task / env to evaluate. Choices: countdown, knights_knaves, "
            "shortest_path, math, kodcode, opencodeinstruct, openmathreasoning."
        ),
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
        help=(
            "Where to write results (default: unique EVAL/... dir including task, "
            "split, model, checkpoint slug, decode caps, and timestamp)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an output dir that already has results.jsonl.",
    )
    parser.add_argument(
        "--exclude-clamped",
        action="store_true",
        help=(
            "Drop examples whose prompt would clamp max_new_tokens (uniform "
            "completion budget). Overflow prompts that cannot fit are always dropped."
        ),
    )
    parser.add_argument(
        "--eos-source",
        type=str,
        default=COMPARISON_EOS_SOURCE,
        choices=("model", "tokenizer", "union"),
        help=(
            f"EOS id source (comparison protocol freezes {COMPARISON_EOS_SOURCE!r}). "
            "Override only for debugging — Base/SFT/GRPO runs must share the same value."
        ),
    )
    parser.add_argument(
        "--expect-eos-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated EOS token ids that must match the resolved set "
            "(e.g. '151645'). Required with --strict-comparison."
        ),
    )
    parser.add_argument(
        "--strict-comparison",
        action="store_true",
        help=(
            "Final Base/SFT/GRPO mode: require --eos-source to match the frozen "
            f"protocol ({COMPARISON_EOS_SOURCE!r}) and require --expect-eos-ids."
        ),
    )
    parser.add_argument(
        "--allow-debug-protocol",
        action="store_true",
        help=(
            "Allow non-protocol --eos-source without --strict-comparison "
            "(development only)."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of eval examples (smoke runs).",
    )
    return parser.parse_args()


def _slug(text: str, max_len: int = 48) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in text)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-_")[:max_len] or "x"


def default_output_dir(args: argparse.Namespace) -> Path:
    """Unique run directory: task/split/model/ckpt/decode + time + pid + nonce."""
    model_slug = _slug(Path(args.model).name if "/" in args.model else args.model)
    ckpt_slug = (
        _slug(Path(args.checkpoint).name) if args.checkpoint else "base"
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    nonce = secrets.token_hex(3)
    name = (
        f"eval-{args.task}-{args.dataset_split}-greedy-"
        f"mt{args.max_new_tokens}-ctx{args.context_tokens}-"
        f"{model_slug}-{ckpt_slug}-{stamp}-p{os.getpid()}-{nonce}"
    )
    return Path("EVAL") / name


def build_dataset_manifest(
    dataset: Dataset,
    task: str,
    *,
    excluded_problem_ids: list[str] | None = None,
    load_stats: dict | None = None,
) -> dict:
    """
    Fingerprint the exact evaluated subset for Base/SFT/GRPO comparability.

    Hash covers problem_id + prompt + grading_reference (answer/entry fields the
    verifier uses). Duplicate problem_ids in the selected set are rejected.
    """
    rows: list[dict] = []
    seen: dict[str, int] = {}
    for i, row in enumerate(dataset):
        row_dict = dict(row) if not isinstance(row, dict) else row
        pid = problem_id_for_row(row_dict, task)
        if pid in seen:
            raise ValueError(
                f"Duplicate problem_id {pid!r} in eval set "
                f"(rows {seen[pid]} and {i}). IDs must be unique."
            )
        seen[pid] = i
        prompt = canonical_prompt(row_dict, task)
        rows.append(
            {
                "problem_id": pid,
                "prompt": prompt,
                "grading_reference": grading_reference_for_row(row_dict, task),
            }
        )
    rows.sort(key=lambda r: r["problem_id"])
    fingerprint = hashlib.sha1(
        json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "n": len(rows),
        "problem_ids": [r["problem_id"] for r in rows],
        "dataset_fingerprint": fingerprint,
        "excluded_problem_ids": sorted(excluded_problem_ids or []),
        "fingerprint_fields": ["problem_id", "prompt", "grading_reference"],
        "load_stats": load_stats or {},
        "problem_ids_unique": True,
    }


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """Create output_dir; refuse to clobber an existing results.jsonl unless overwrite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    if results_path.exists() and results_path.stat().st_size > 0 and not overwrite:
        raise FileExistsError(
            f"{results_path} already exists. Pass --overwrite to replace, "
            "or choose a new --output-dir (defaults are unique per run)."
        )


def build_run_metadata(
    args: argparse.Namespace,
    *,
    config: dict,
    num_examples: int,
    output_dir: Path,
    clamp_survey: dict | None = None,
    dataset_manifest: dict | None = None,
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
        "exclude_clamped": bool(args.exclude_clamped),
        "model": args.model,
        "checkpoint": checkpoint,
        "checkpoint_resolved": str(Path(checkpoint).resolve()) if checkpoint else None,
        "output_dir": str(output_dir.resolve()),
        "context_clamping": clamp_survey,
        "dataset_manifest": dataset_manifest,
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
            "eos_source": args.eos_source,
        },
        "comparison_protocol": {
            "eos_source": COMPARISON_EOS_SOURCE,
            "do_sample": False,
            "num_return_sequences": 1,
            "max_new_tokens_default": MAX_COMPLETION_TOKENS,
            "context_tokens_default": CONTEXT_TOKENS,
            "eos_source_matches_protocol": args.eos_source == COMPARISON_EOS_SOURCE,
            # Smoke Base then lock with --expect-eos-ids so SFT/GRPO must match.
            "effective_eos_token_ids": (config.get("eos_config") or {}).get(
                "eos_token_ids"
            ),
            "expect_eos_ids": args.expect_eos_ids,
        },
        "defaults": {
            "MODEL_ID": MODEL_ID,
            "MAX_COMPLETION_TOKENS": MAX_COMPLETION_TOKENS,
            "CONTEXT_TOKENS": CONTEXT_TOKENS,
            "COMPARISON_EOS_SOURCE": COMPARISON_EOS_SOURCE,
        },
        "cli": vars(args),
    }


def main() -> None:
    """CLI entry: load data + policy → evaluate → write metadata/results/summary."""
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    ensure_output_dir(output_dir, overwrite=args.overwrite)

    # Need at least one token of headroom for the prompt inside the context window.
    if args.max_new_tokens >= args.context_tokens:
        raise ValueError(
            f"--max-new-tokens ({args.max_new_tokens}) must be < "
            f"--context-tokens ({args.context_tokens}) so prompts can fit"
        )

    # Decode protocol knobs passed into generate_response (also copied to metadata).
    config = {
        "max_new_tokens": args.max_new_tokens,
        "context_tokens": args.context_tokens,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "stop_sequences": None,
        "inherits_model_generation_config": False,
    }

    dataset, load_stats = load_examples(args.dataset, args.dataset_split, args.task)
    if args.max_examples is not None:
        n = min(args.max_examples, len(dataset))
        dataset = dataset.select(range(n))
        load_stats = {
            **load_stats,
            "n_after_max_examples": len(dataset),
            "max_examples": args.max_examples,
        }

    model, tokenizer = load_policy(args.model, checkpoint=args.checkpoint)

    if args.strict_comparison:
        if args.eos_source != COMPARISON_EOS_SOURCE:
            raise ValueError(
                f"--strict-comparison requires --eos-source={COMPARISON_EOS_SOURCE!r} "
                f"(got {args.eos_source!r})"
            )
        if not args.expect_eos_ids:
            raise ValueError(
                "--strict-comparison requires --expect-eos-ids "
                "(lock IDs from a base-model smoke, then reuse for SFT/GRPO)"
            )
    elif args.eos_source != COMPARISON_EOS_SOURCE and not args.allow_debug_protocol:
        raise ValueError(
            f"--eos-source={args.eos_source!r} differs from frozen protocol "
            f"{COMPARISON_EOS_SOURCE!r}. Pass --allow-debug-protocol for development "
            "smokes, or use --strict-comparison with --expect-eos-ids for final runs."
        )

    # EOS ids: frozen comparison default is model.generation_config (not silent union).
    eos_config = resolve_eos_config(model, tokenizer, source=args.eos_source)
    config["eos_config"] = eos_config
    config["eos_token_id"] = (
        eos_config["eos_token_ids"][0] if eos_config["eos_token_ids"] else None
    )
    config["eos_token_ids"] = eos_config["eos_token_ids"]
    config["pad_token_id"] = eos_config["pad_token_id"]
    config["generation_settings"] = build_eval_generation_settings(
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        context_tokens=args.context_tokens,
        eos_config=eos_config,
    )
    print(
        f"EOS config: source={eos_config['eos_source_resolved']} "
        f"tokenizer={eos_config['tokenizer_eos_token_id']!r} "
        f"model_gen={eos_config['model_generation_config_eos_token_id']!r} "
        f"effective={eos_config['eos_token_ids']}"
    )
    if args.expect_eos_ids is not None:
        expected = sorted(
            int(x.strip()) for x in args.expect_eos_ids.split(",") if x.strip()
        )
        actual = list(eos_config["eos_token_ids"])
        if actual != expected:
            raise ValueError(
                f"Effective EOS ids {actual} != --expect-eos-ids {expected}. "
                "Base/SFT/GRPO comparison requires identical stop tokens."
            )
        print(f"EOS ids locked to expect-eos-ids={expected}")

    # Survey clamping before generation; optionally drop uneven budgets.
    clamp_survey = survey_context_clamping(
        dataset,
        args.task,
        tokenizer,
        args.max_new_tokens,
        args.context_tokens,
    )
    print(
        f"context clamp survey: fit={clamp_survey['n_fit_full_budget']} "
        f"clamped={clamp_survey['n_clamped']} "
        f"overflow={clamp_survey['n_prompt_overflow']}"
    )
    drop_ids = set(clamp_survey["overflow_problem_ids"])
    if args.exclude_clamped:
        drop_ids |= set(clamp_survey["clamped_problem_ids"])
    if drop_ids:
        keep_indices = [
            i
            for i, row in enumerate(dataset)
            if problem_id_for_row(dict(row) if not isinstance(row, dict) else row, args.task)
            not in drop_ids
        ]
        dataset = dataset.select(keep_indices)
        print(f"excluded {len(drop_ids)} problem_ids → rows={len(dataset)}")

    if len(dataset) == 0:
        raise ValueError(
            "Evaluation set is empty after loading/filtering "
            "(check --max-examples / --exclude-clamped / context overflow). "
            "Empty runs are rejected."
        )

    dataset_manifest = build_dataset_manifest(
        dataset,
        args.task,
        excluded_problem_ids=sorted(drop_ids),
        load_stats={
            **load_stats,
            "n_final_eval": len(dataset),
            "n_excluded_context": len(drop_ids),
        },
    )
    expected_n = len(dataset)

    print(
        f"eval task={args.task} | raw={load_stats.get('n_raw')} "
        f"eligible={load_stats.get('n_after_eligibility_filters')} "
        f"final={expected_n} | "
        f"fingerprint={dataset_manifest['dataset_fingerprint'][:12]}… | "
        f"checkpoint={args.checkpoint!r} | "
        f"max_new_tokens={args.max_new_tokens} | context_tokens={args.context_tokens} | "
        f"output_dir={output_dir}"
    )

    # Write metadata first so a crash mid-eval still leaves run identity on disk.
    metadata = build_run_metadata(
        args,
        config=config,
        num_examples=expected_n,
        output_dir=output_dir,
        clamp_survey=clamp_survey,
        dataset_manifest=dataset_manifest,
    )
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(dataset_manifest, indent=2) + "\n", encoding="utf-8"
    )

    results_path = output_dir / "results.jsonl"
    if args.overwrite and results_path.exists():
        results_path.unlink()

    results = evaluate_dataset(
        model,
        tokenizer,
        dataset,
        args.task,
        config,
        results_path=results_path,
    )
    if len(results) != expected_n:
        raise RuntimeError(
            f"Result count mismatch: expected {expected_n} records, got {len(results)}"
        )

    summary = summarize_results(results)
    summary["metadata"] = metadata
    summary["dataset_fingerprint"] = dataset_manifest["dataset_fingerprint"]

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {metadata_path}, {manifest_path}, {results_path}, and {summary_path}")

    # Persist artifacts first; fail the process if incomplete so Slurm does not
    # treat a partial eval as a successful job.
    if not summary["complete"]:
        print(
            f"[eval] INCOMPLETE: n={summary['n']} n_scored={summary['n_scored']} "
            f"n_infra_errors={summary['n_infra_errors']}; "
            f"accuracy={summary['accuracy']} "
            "(exiting nonzero — not a comparable final benchmark).",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()




