"""Evaluate fixed test questions, save raw outputs, and resume matching runs."""
from __future__ import annotations

import copy
import csv
import importlib.metadata
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .data import TASKS, load_test
from .graders import TaskGrader, reference_text
from .io_utils import append_jsonl, file_hash, fingerprint, read_journal, read_json, write_json


def is_local_source(root, source):
    return (root / source).exists() or source.startswith(("/", "./", "../", "checkpoints/", "checkpoints\\")) or ":\\" in source


def resolve_config(root, config):
    manifest = read_json(root / "data/manifest.json")
    config = copy.deepcopy(config)
    generation = config.setdefault("generation", {})
    generation.setdefault("max_new_tokens", 4096)
    generation.setdefault("seed", 42)
    if set(generation) - {"max_new_tokens", "seed"}:
        raise ValueError("Generation supports max_new_tokens and seed only; protocol is one greedy completion per question")
    if type(generation["max_new_tokens"]) is not int or generation["max_new_tokens"] <= 0 or type(generation["seed"]) is not int:
        raise ValueError("Generation budget must be a positive integer and seed must be an integer")
    runs = config.get("runs", [])
    if not runs:
        raise ValueError("Config must contain at least one completed-checkpoint run")
    names = set()
    for run in runs:
        name = run.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
            raise ValueError("Run names must be unique and contain letters, numbers, dots, underscores or hyphens")
        names.add(name)
        if run.get("task") not in TASKS:
            raise ValueError(f"Choose a task from {sorted(TASKS)}")
        run.setdefault("data", {key: manifest[run["task"]][key] for key in ("path", "rows", "sha256")})
        data = run["data"]
        if type(data.get("rows")) is not int or data["rows"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", data.get("sha256", "")):
            raise ValueError("Every test file requires its agreed row count and SHA-256")
        checkpoint = run.get("checkpoint", {})
        if checkpoint.get("type") not in {"adapter", "full"} or not checkpoint.get("path"):
            raise ValueError("checkpoint needs type=adapter/full and path")
        if checkpoint.get("loader") not in {"auto_causal", "qwen3_5_text", "qwen3_5_full"}:
            raise ValueError("Choose the same checkpoint architecture: auto_causal, qwen3_5_text or qwen3_5_full")
        source = checkpoint.get("base_model") if checkpoint["type"] == "adapter" else checkpoint["path"]
        revision = checkpoint.get("base_revision") if checkpoint["type"] == "adapter" else checkpoint.get("revision")
        if not source:
            raise ValueError("Adapter evaluation requires base_model")
        if not is_local_source(root, source) and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError("Pin remote model/base_revision to its exact 40-character commit SHA")
        prompt = run.get("prompt", {})
        if prompt.get("format") not in {"base", "chat", "raw"}:
            raise ValueError("Explicitly choose prompt.format=base/chat/raw")
        if prompt["format"] == "base":
            template = prompt.get("base_template", "")
            if "{question}" not in template or set(re.findall(r"\{([^{}]*)\}", template)) - {"question", "system"}:
                raise ValueError("Base template must include {question}; only {question} and {system} are supported")
        if run.get("tokenizer") and not is_local_source(root, run["tokenizer"]) and not re.fullmatch(r"[0-9a-f]{40}", run.get("tokenizer_revision", "")):
            raise ValueError("Pin an explicitly configured remote tokenizer_revision to its exact commit SHA")
        if prompt["format"] == "chat" and "qwen" in source.lower() and "enable_thinking" not in prompt.get("chat_template_kwargs", {}):
            raise ValueError("Set Qwen chat_template_kwargs.enable_thinking explicitly for this comparison")
        if run.get("device", "cuda").split(":")[0] not in {"cpu", "cuda"} or run.get("dtype", "bfloat16") not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Supported devices: cuda/cuda:N/cpu; dtypes: float32/float16/bfloat16")
        if run["task"] in {"kodcode", "opencode"}:
            run.setdefault("grader", config.get("grader", {}))
            if run["grader"].get("runtime") not in {"docker", "apptainer"} or not run["grader"].get("image"):
                raise ValueError("Coding tasks require a docker/apptainer grader image")
    return config


def checkpoint_identity(root, run):
    """Hash local inference files so changed checkpoints cannot reuse old outputs."""
    checkpoint = run["checkpoint"]
    paths = [checkpoint["path"]]
    if checkpoint["type"] == "adapter":
        paths.append(checkpoint["base_model"])
    if run.get("tokenizer"):
        paths.append(run["tokenizer"])
    identities = {}
    for value in paths:
        path = (root / value).resolve()
        if not path.exists():
            if (value == checkpoint["path"] and checkpoint["type"] == "adapter") or is_local_source(root, value):
                raise FileNotFoundError(f"Checkpoint/tokenizer folder is missing: {path}; set the correct path in configs/eval.json")
            continue  # Remote sources are pinned by commit SHA in the run config.
        if not path.is_dir():
            raise ValueError(f"Model/tokenizer path must be a saved checkpoint folder: {path}")
        if value == checkpoint["path"]:
            if checkpoint["type"] == "adapter":
                if not (path / "adapter_config.json").is_file() or not any(path.glob("adapter_model.*")):
                    raise ValueError("Adapter needs adapter_config.json and adapter_model.safetensors/bin")
            elif (path / "adapter_config.json").exists():
                raise ValueError("This folder is an adapter; configure type=adapter and specify its original base")
            elif not (path / "config.json").is_file() or not any(path.glob("*.safetensors")) and not any(path.glob("pytorch_model*.bin")):
                raise ValueError("A full checkpoint requires config.json and complete model weights")
        files = [p for p in path.iterdir() if p.is_file() and (
            p.suffix in {".json", ".safetensors", ".model", ".tiktoken"} or p.name.startswith(("pytorch_model", "adapter_model", "vocab", "merges"))
        ) and not p.name.startswith(("optimizer", "scheduler", "trainer_state"))]
        identities[value] = {p.name: file_hash(p) for p in sorted(files)}
    return identities


def environment():
    import platform
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "transformers", "peft", "accelerate", "safetensors", "math-verify", "latex2sympy2_extended"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def checked_records(path, question_ids, contract):
    records = read_journal(path)
    if len(records) > len(question_ids):
        raise ValueError(f"Too many records in {path}")
    for index, record in enumerate(records):
        if record.get("question_id") != question_ids[index] or record.get("run_fingerprint") != contract:
            raise ValueError(f"Saved results do not match the current run: {path}; use a new run name")
    return records


def summary(run, rows, predictions, scores, smoke):
    n, requested = len(scores), len(rows)
    correct = sum(row["correct"] for row in scores)
    complete = n == requested
    return {
        "run": run["name"], "task": run["task"], "split": "test", "smoke_test": smoke,
        "test_sha256": run["data"]["sha256"], "requested": requested, "generated": len(predictions),
        "scored": n, "complete": complete, "correct": correct,
        "pass_at_1": correct / requested if complete else None,
        "accuracy_percent": 100 * correct / requested if complete else None,
        "accuracy_on_completed": correct / n if n else None,
        "mean_task_score": sum(r["score"] for r in scores) / n if n else None,
        "failure_types": dict(Counter(r["status"] for r in scores if not r["correct"])),
        "token_limit_hits": sum(p.get("hit_token_limit", False) for p in predictions),
        "unfinished_reasoning": sum(p.get("unfinished_reasoning", False) for p in predictions),
        "generated_tokens": sum(p.get("generated_tokens", 0) for p in predictions),
        "generation_seconds": sum(p.get("generation_seconds", 0) for p in predictions),
        "note": "One greedy completion per question. All fixed test rows remain in the denominator. For Shortest Path, score 0.5 is a failure for pass@1. Smoke results are not final test results.",
    }


def run_one(root, output_root, run, generation, rows, limit, stage):
    output = output_root / run["name"] / (f"smoke-{limit}" if limit else "test")
    if stage == "references":
        output = output / "reference-check"
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(p.relative_to(root)): file_hash(p) for p in sorted((root / "src").glob("*.py"))}
    source_hashes["run_all.py"] = file_hash(root / "run_all.py")
    checkpoint_hashes = {} if stage == "references" else checkpoint_identity(root, run)
    contract = fingerprint({"run": run, "generation": generation, "code": source_hashes,
                            "checkpoint": checkpoint_hashes, "limit": limit})
    contract_path = output / "run_config.json"
    if contract_path.exists() and read_json(contract_path)["run_fingerprint"] != contract:
        raise ValueError(f"Checkpoint, settings or code changed in {output}; use a new run name to preserve prior results")
    write_json(contract_path, {"run_fingerprint": contract, "run": run, "generation": generation,
                               "limit": limit, "source_sha256": source_hashes, "checkpoint_sha256": checkpoint_hashes,
                               "created_at": datetime.now(timezone.utc).isoformat()})
    question_ids = [r["question_id"] for r in rows]
    if stage == "references":
        saved = checked_records(output / "reference_scores.jsonl", question_ids, contract)
        grader = TaskGrader(root, run)
        try:
            for row in rows[len(saved):]:
                score = {"question_id": row["question_id"], "run_fingerprint": contract,
                         **grader.score(row, reference_text(run["task"], row))}
                append_jsonl(output / "reference_scores.jsonl", score)
                saved.append(score)
                print(f"{run['name']} reference tests {len(saved)}/{len(rows)}", flush=True)
        finally:
            grader.close()
        write_json(output / "reference_summary.json", {"tested": len(saved), "passing": sum(r["correct"] for r in saved),
                   "failures": [{"question_id": r["question_id"], "status": r["status"]} for r in saved if not r["correct"]],
                   "note": "Reference failures are reported for review; they do not remove questions from the main denominator."})
        return None
    predictions = checked_records(output / "predictions.jsonl", question_ids, contract)
    scores = checked_records(output / "scores.jsonl", question_ids, contract)
    if len(scores) > len(predictions):
        raise ValueError("Scores exist without their original predictions")
    for index, score in enumerate(scores):
        if score["prediction_sha256"] != fingerprint(predictions[index]):
            raise ValueError("Saved prediction content differs from the content that was graded")
    grader = None
    try:
        if stage in {"all", "grade"} and len(scores) < len(rows):
            grader = TaskGrader(root, run)
            grading_info = {"environment": environment(), "grader": grader.identity}
            grading_path = output / "grading_environment.json"
            if grading_path.exists() and read_json(grading_path) != grading_info and scores:
                raise ValueError("Grader environment changed; use a new run name to avoid mixing scores")
            write_json(grading_path, grading_info)
        def score_pending():
            if grader is not None:
                for index in range(len(scores), len(predictions)):
                    prediction = predictions[index]
                    score = {"question_id": rows[index]["question_id"], "run_fingerprint": contract,
                             "prediction_sha256": fingerprint(prediction), **grader.score(rows[index], prediction["final_text"])}
                    append_jsonl(output / "scores.jsonl", score)
                    scores.append(score)
                    print(f"{run['name']} test {len(scores)}/{len(rows)} | correct {sum(s['correct'] for s in scores)}", flush=True)
        score_pending()
        if stage in {"all", "generate"} and len(predictions) < len(rows):
            from transformers import set_seed
            from .model_loader import ModelRunner
            set_seed(generation["seed"])
            runner = ModelRunner(root, run, generation)
            try:
                write_json(output / "environment.json", environment())
                write_json(output / "model_load.json", runner.metadata)
                for row in rows[len(predictions):]:
                    prediction = {"run_fingerprint": contract, **runner.generate(row)}
                    append_jsonl(output / "predictions.jsonl", prediction)
                    predictions.append(prediction)
                    score_pending()
                    if stage == "generate":
                        print(f"{run['name']} generated {len(predictions)}/{len(rows)}", flush=True)
                    write_json(output / "summary.json", summary(run, rows, predictions, scores, bool(limit)))
            finally:
                runner.close()
        if stage == "grade" and len(predictions) != len(rows):
            raise ValueError(f"Only {len(predictions)}/{len(rows)} predictions exist; run generation before final grading")
        result = summary(run, rows, predictions, scores, bool(limit))
        write_json(output / "summary.json", result)
        print(f"Saved {output / 'summary.json'}", flush=True)
        return result
    finally:
        if grader is not None:
            grader.close()


def run_all(config_path, *, names=None, limit=None, check_only=False, stage="all"):
    """Public entry point: load config, then evaluate each completed checkpoint."""
    root = Path(__file__).resolve().parents[1]
    config_path = Path(config_path).resolve()
    config = resolve_config(root, read_json(config_path))
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("--limit must be a positive integer")
    runs = [r for r in config["runs"] if names is None or r["name"] in names]
    if names and set(names) - {r["name"] for r in runs}:
        raise ValueError("Unknown --run name")
    loaded = []
    # Check all selected test files before starting GPU work.
    for run in runs:
        rows = load_test(root, run)
        if limit and limit > len(rows):
            raise ValueError("--limit exceeds the fixed test set size")
        selected = rows[:limit] if limit else rows
        loaded.append((run, selected))
        print(f"{run['name']}: test {len(rows)} rows | SHA-256 verified | requested {len(selected)}", flush=True)
    if check_only:
        print("Config and test data checks passed. Model loading and grader runtime have not been exercised.")
        return []
    output_root = (root / config.get("output_dir", "outputs")).resolve()
    results = []
    for run, rows in loaded:
        result = run_one(root, output_root, run, config["generation"], rows, limit, stage)
        if result is not None:
            results.append(result)
    if results:
        path = output_root / (f"smoke-{limit}-summary.csv" if limit else "test-summary.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = ["run", "task", "requested", "generated", "scored", "complete", "correct", "pass_at_1", "accuracy_percent", "token_limit_hits", "generation_seconds", "test_sha256"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    return results
