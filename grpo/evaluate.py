"""
Evaluate the base model (baselines) or GRPO/SFT LoRA checkpoints on a task split,
with the SAME prompts, scorer and token caps as training (owner: Mehul).

Why it reuses training code
  prompts   built by grpo.grpo.load_train_dataset (_normalize_row -> env.load_problem),
            so eval prompts are byte-identical to training prompts.
  scorer    env.step(completion) from the same env classes (strict final <answer>
            rule in grpo/answer_extract.py); unscored (None) counts as reward 0,
            exactly like make_env_reward_fn.
  sampling  temperature 1.0, top_p 1.0, no top_k/min_p, max_tokens = completion cap,
            vLLM max_model_len = context cap (grpo.grpo defaults 4096 / 5120).
  adapters  merged and pushed into vLLM the way TRL's colocate weight sync does
            (incl. the Qwen3.5 text->multimodal name mapping from grpo.grpo), so a
            checkpoint is evaluated through the same weight path it was trained with.

What it produces, per split
  greedy    accuracy (accept rate) and mean reward of one greedy completion
  sampled   k completions at temperature 1.0 -> per-prompt solve rate p,
            pass@{1,2,4,8,16}, difficulty bucket (Hard p<0.2, Intermediate,
            Easy p>0.8), and predicted GRPO group outcomes for G in {2,4,8,16}
            (expected all-accepted / none-accepted / zero-reward-std groups,
            estimated without replacement from the k samples)

Files, under --output-dir (default baselines/<model>/<task> for the base model,
<adapter>/eval/<task> for a checkpoint):

    <split>[_n<limit>].json               summary + full provenance manifest
    <split>[_n<limit>]_per_prompt.jsonl   one row per prompt (rewards for all k samples)
    <split>[_n<limit>]_completions.jsonl  texts (--save-completions greedy|all|none)

Examples
    # baselines: greedy on all of val/test, k=16 on a fixed 500-prompt train sample
    python -m grpo.evaluate --task math --dataset data_splits/competition_math \
        --splits val,test,train:500 --k 16
    # several checkpoints, one vLLM start-up
    python -m grpo.evaluate --task shortest_path --dataset data/shortest_path \
        --splits val --k 0 --adapter runs/X/checkpoint-50 --adapter runs/X/checkpoint-100
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TASKS = ("countdown", "knights_knaves", "shortest_path", "math", "kodcode", "opencodeinstruct")
CODE_TASKS = ("kodcode", "opencodeinstruct")   # verifier runs tests in subprocesses
DEFAULT_COMPLETION_TOKENS = 4096              # grpo.grpo.MAX_COMPLETION_TOKENS
DEFAULT_CONTEXT_TOKENS = 5120                 # grpo.grpo.VLLM_CONTEXT_TOKENS
HARD_BELOW, EASY_ABOVE = 0.2, 0.8
PASS_AT = (1, 2, 4, 8, 16)
GROUP_SIZES = (2, 4, 8, 16)


# ---------------------------------------------------------------------------
# Pure metric helpers (no heavy deps; unit-tested)
# ---------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al. 2021)."""
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def prob_group_all_equal(values: Sequence[Any], G: int) -> float:
    """P(G draws without replacement from `values` are all identical)."""
    n = len(values)
    if G > n:
        raise ValueError(f"G={G} > n={n}")
    return sum(comb(c, G) for c in Counter(values).values()) / comb(n, G)


def bucket(p: float | None) -> str | None:
    if p is None:
        return None
    if p < HARD_BELOW:
        return "hard"
    if p > EASY_ABOVE:
        return "easy"
    return "intermediate"


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def per_prompt_record(idx: int, prompt: str, greedy: dict | None,
                      samples: list[dict]) -> dict[str, Any]:
    """Collapse one prompt's scored generations into the per-prompt row."""
    rec: dict[str, Any] = {"idx": idx, "prompt_hash": prompt_hash(prompt)}
    if greedy is not None:
        rec.update({
            "greedy_reward": greedy["reward"], "greedy_accepted": greedy["accepted"],
            "greedy_fail": greedy["fail"], "greedy_n_tokens": greedy["n_tokens"],
            "greedy_hit_length": greedy["hit_length"],
        })
    k = len(samples)
    rec["k"] = k
    if k:
        acc = [s["accepted"] for s in samples]
        rew = [s["reward"] for s in samples]
        p = sum(acc) / k
        rec.update({
            "n_accepted": sum(acc), "p_accept": p, "mean_reward": _mean(rew),
            "rewards": rew, "accepted": acc,
            "fails": dict(Counter(s["fail"] for s in samples if s["fail"])),
            "mean_n_tokens": _mean([s["n_tokens"] for s in samples]),
            "hit_length_rate": sum(s["hit_length"] for s in samples) / k,
            "bucket": bucket(p),
        })
    return rec


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Split-level metrics from per-prompt rows."""
    out: dict[str, Any] = {"n_prompts": len(records)}
    g = [r for r in records if "greedy_reward" in r]
    if g:
        n = len(g)
        out["greedy"] = {
            "accuracy": sum(r["greedy_accepted"] for r in g) / n,
            "mean_reward": _mean([r["greedy_reward"] for r in g]),
            "fail_rates": {f: c / n for f, c in
                           Counter(r["greedy_fail"] for r in g if r["greedy_fail"]).items()},
            "mean_n_tokens": _mean([r["greedy_n_tokens"] for r in g]),
            "hit_length_rate": sum(r["greedy_hit_length"] for r in g) / n,
        }
    s = [r for r in records if r.get("k")]
    if s:
        k = min(r["k"] for r in s)
        n_samples = sum(r["k"] for r in s)
        fails: Counter[str] = Counter()
        for r in s:
            fails.update(r["fails"])
        buckets = Counter(r["bucket"] for r in s)
        sampled: dict[str, Any] = {
            "k": k,
            "pass_at": {str(j): _mean([pass_at_k(r["k"], r["n_accepted"], j) for r in s])
                        for j in PASS_AT if j <= k},
            "mean_reward": _mean([r["mean_reward"] for r in s]),
            "mean_p_accept": _mean([r["p_accept"] for r in s]),
            "fail_rates": {f: c / n_samples for f, c in fails.items()},
            "mean_n_tokens": _mean([r["mean_n_tokens"] for r in s]),
            "hit_length_rate": _mean([r["hit_length_rate"] for r in s]),
            "buckets": {b: buckets.get(b, 0) / len(s) for b in ("hard", "intermediate", "easy")},
            "bucket_counts": {b: buckets.get(b, 0) for b in ("hard", "intermediate", "easy")},
        }
        # What GRPO would see at each G if it trained on these prompts with this policy.
        pred = {}
        for G in GROUP_SIZES:
            if G > k:
                continue
            all_acc = _mean([comb(r["n_accepted"], G) / comb(r["k"], G) for r in s])
            none_acc = _mean([comb(r["k"] - r["n_accepted"], G) / comb(r["k"], G) for r in s])
            pred[str(G)] = {
                "frac_all_accepted": all_acc,
                "frac_none_accepted": none_acc,
                "frac_mixed_acceptance": 1.0 - all_acc - none_acc,
                "frac_zero_std": _mean([prob_group_all_equal(r["rewards"], G) for r in s]),
            }
        sampled["group_predictions"] = pred
        out["sampled"] = sampled
    return out


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def parse_split_specs(spec: str) -> list[tuple[str, int | None]]:
    """'val,test,train:500' -> [('val', None), ('test', None), ('train', 500)]"""
    out = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, _, lim = part.partition(":")
        out.append((name, int(lim) if lim else None))
    if not out:
        raise ValueError("no splits given")
    return out


def sample_indices(n: int, limit: int | None, seed: int) -> list[int]:
    if limit is None or limit >= n:
        return list(range(n))
    return sorted(random.Random(seed).sample(range(n), limit))


# ---------------------------------------------------------------------------
# Data + scoring (reuse training code)
# ---------------------------------------------------------------------------
@dataclass
class Example:
    idx: int          # row index in the (filtered, normalized) split
    prompt: str
    entry: dict[str, Any]


def resolve_split_file(source: str, split: str) -> Path | None:
    p = Path(source)
    if p.is_file():
        return p
    if p.is_dir():
        for ext in ("parquet", "jsonl", "json"):
            if (p / f"{split}.{ext}").exists():
                return p / f"{split}.{ext}"
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_env_factory(task: str) -> Callable[[], Any]:
    import grpo.grpo as g
    return lambda: g.make_env_for_task(task)


def load_examples(source: str, split: str, task: str) -> list[Example]:
    import grpo.grpo as g
    ds = g.load_train_dataset(source, split, task=task)
    return [Example(i, p, e) for i, (p, e) in enumerate(zip(ds["prompt"], ds["entry"]))]


class Scorer:
    """env.step() per completion; thread-local envs so code-task tests can run in parallel."""

    def __init__(self, task: str, workers: int = 1) -> None:
        from grpo.metrics import _failure_type
        self._failure_type = _failure_type
        self.make_env = make_env_factory(task)
        # math_verify uses signal-based timeouts (main thread only): parallelize code tasks only.
        self.workers = workers if task in CODE_TASKS else 1
        self._local = threading.local()

    def _env(self):
        if not hasattr(self._local, "env"):
            self._local.env = self.make_env()
        return self._local.env

    def score_one(self, entry: dict[str, Any], gen: dict[str, Any]) -> dict[str, Any]:
        env = self._env()
        env.load_problem(entry)
        out = env.step(gen["text"])
        reward = out.get("reward")
        return {
            "reward": 0.0 if reward is None else float(reward),
            "accepted": bool(out.get("acceptor")),
            "fail": self._failure_type(out),
            "unscored": reward is None,
            "n_tokens": gen["n_tokens"], "hit_length": gen["hit_length"],
        }

    def score(self, pairs: list[tuple[dict, dict]]) -> list[dict[str, Any]]:
        if self.workers <= 1:
            return [self.score_one(e, g) for e, g in pairs]
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda eg: self.score_one(*eg), pairs))


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _vllm_param_name(name: str) -> str:
    """Same name rewriting as TRL colocate sync + grpo.grpo's Qwen3.5 patch."""
    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
    name = name.replace("_checkpoint_wrapped_module.", "").replace("modules_to_save.default.", "")
    return name


class VLLMBackend:
    """Colocate-style vLLM engine; adapters are merged and pushed like TRL's weight sync."""

    name = "vllm"

    def __init__(self, model_id: str, context_tokens: int, seed: int,
                 gpu_memory_utilization: float) -> None:
        # Same in-process engine TRL uses in colocate mode (needed for load_weights).
        for k, v in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"),
                     ("MASTER_ADDR", "localhost"), ("MASTER_PORT", str(_free_port()))):
            os.environ.setdefault(k, v)
        from vllm import LLM
        self.model_id = model_id
        self.llm = LLM(model=model_id, max_model_len=context_tokens, dtype="bfloat16",
                       gpu_memory_utilization=gpu_memory_utilization, seed=seed,
                       distributed_executor_backend="external_launcher")
        self.weights = "base"

    def _push(self, named_params) -> None:
        model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        for name, param in named_params:
            model.load_weights([(name, param.to("cuda"))])
        self.llm.reset_prefix_cache()

    def _named_params(self, adapter: str | None):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        import grpo.grpo as g
        base = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        qwen35 = getattr(base.config, "model_type", None) in g.QWEN35_MODEL_TYPES
        model = base
        lora_prefix = "lora_"
        if adapter:
            model = PeftModel.from_pretrained(base, adapter)
            model.merge_adapter()
            lora_prefix = getattr(model, "prefix", "lora_")
        for name, param in model.named_parameters():
            if adapter and (lora_prefix in name or "original_module" in name):
                continue
            name = _vllm_param_name(name)
            if qwen35:
                for prefix, repl in g.QWEN35_VLLM_PREFIX_MAP:
                    if name.startswith(prefix):
                        name = name.replace(prefix, repl, 1)
                        break
            yield name, param.data

    def load_adapter(self, adapter: str | None) -> None:
        want = adapter or "base"
        if want == self.weights:
            return
        self._push(self._named_params(adapter))
        self.weights = want

    def generate(self, prompts: list[str], n: int, temperature: float,
                 max_tokens: int, seed: int) -> list[list[dict[str, Any]]]:
        from vllm import SamplingParams
        sp = SamplingParams(n=n, temperature=temperature, top_p=1.0,
                            repetition_penalty=1.0, max_tokens=max_tokens, seed=seed)
        outs = self.llm.generate(prompts, sp, use_tqdm=True)
        return [[{"text": o.text, "n_tokens": len(o.token_ids),
                  "hit_length": o.finish_reason == "length"} for o in out.outputs]
                for out in outs]


class HFBackend:
    """transformers.generate fallback (CPU smoke tests, or no vLLM)."""

    name = "hf"

    def __init__(self, model_id: str, context_tokens: int, seed: int,
                 batch_size: int = 8) -> None:
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.model_id, self.context_tokens, self.batch_size = model_id, context_tokens, batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.weights = None

    def load_adapter(self, adapter: str | None) -> None:
        want = adapter or "base"
        if want == self.weights:
            return
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        dtype = self.torch.bfloat16 if self.device == "cuda" else self.torch.float32
        model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype)
        if adapter:
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.weights = want

    def generate(self, prompts: list[str], n: int, temperature: float,
                 max_tokens: int, seed: int) -> list[list[dict[str, Any]]]:
        torch = self.torch
        torch.manual_seed(seed)
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos if eos is not None
                                                          else self.tok.eos_token_id])
        results = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True).to(self.device)
            plen = enc["input_ids"].shape[1]
            max_new = max(1, min(max_tokens, self.context_tokens - plen))
            kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) \
                if temperature > 0 else dict(do_sample=False)
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new, num_return_sequences=n,
                                          pad_token_id=self.tok.pad_token_id, **kw)
            gen = out[:, plen:].tolist()
            for b in range(len(batch)):
                row = []
                for ids in gen[b * n:(b + 1) * n]:
                    cut = next((j + 1 for j, t in enumerate(ids) if t in eos_ids), None)
                    ids = ids[:cut] if cut else ids
                    row.append({"text": self.tok.decode(ids, skip_special_tokens=True),
                                "n_tokens": len(ids),
                                "hit_length": cut is None and len(ids) >= max_new})
                results.append(row)
        return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parents[1],
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "trl", "peft", "vllm", "reasoning_gym", "math_verify"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


def adapter_info(adapter: str | None) -> dict[str, Any]:
    if not adapter:
        return {"adapter": None}
    p = Path(adapter)
    info: dict[str, Any] = {"adapter": str(p)}
    weights = [w for w in (p / "adapter_model.safetensors", p / "adapter_model.bin") if w.is_file()]
    if not weights:
        raise FileNotFoundError(f"no adapter_model.* in {p} (is this a LoRA adapter/checkpoint dir?)")
    info["adapter_sha256"] = sha256_file(weights[0])
    ts = p / "trainer_state.json"
    if ts.is_file():
        st = json.loads(ts.read_text())
        info["global_step"] = st.get("global_step")
        info["num_input_tokens_seen"] = st.get("num_input_tokens_seen")
    return info


def default_output_dir(model_id: str, task: str, adapter: str | None) -> Path:
    if adapter:
        return Path(adapter) / "eval" / task
    return Path("baselines") / model_id.rstrip("/").split("/")[-1] / task


def evaluate_split(backend, scorer: Scorer, examples: list[Example], *, k: int, greedy: bool,
                   temperature: float, max_tokens: int, seed: int,
                   save_completions: str) -> tuple[list[dict], list[dict]]:
    prompts = [ex.prompt for ex in examples]
    greedy_scored: list[dict | None] = [None] * len(examples)
    sampled_scored: list[list[dict]] = [[] for _ in examples]
    completions: list[dict] = []

    if greedy:
        gens = backend.generate(prompts, 1, 0.0, max_tokens, seed)
        scored = scorer.score([(ex.entry, g[0]) for ex, g in zip(examples, gens)])
        for i, (ex, g, s) in enumerate(zip(examples, gens, scored)):
            greedy_scored[i] = s
            if save_completions in ("greedy", "all"):
                completions.append({"idx": ex.idx, "prompt_hash": prompt_hash(ex.prompt),
                                    "kind": "greedy", "j": 0, "completion": g[0]["text"], **s})
    if k > 0:
        gens = backend.generate(prompts, k, temperature, max_tokens, seed)
        pairs = [(ex.entry, gj) for ex, g in zip(examples, gens) for gj in g]
        scored = scorer.score(pairs)
        it = iter(scored)
        for i, (ex, g) in enumerate(zip(examples, gens)):
            for j, gj in enumerate(g):
                s = next(it)
                sampled_scored[i].append(s)
                if save_completions == "all":
                    completions.append({"idx": ex.idx, "prompt_hash": prompt_hash(ex.prompt),
                                        "kind": "sample", "j": j, "completion": gj["text"], **s})
    records = [per_prompt_record(ex.idx, ex.prompt, greedy_scored[i], sampled_scored[i])
               for i, ex in enumerate(examples)]
    return records, completions


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--dataset", required=True, help="dir with <split>.{parquet,jsonl,json}, file, or HF id")
    p.add_argument("--splits", default="val", help="comma list, optional :limit, e.g. val,test,train:500")
    p.add_argument("--adapter", action="append", default=None,
                   help="LoRA adapter / checkpoint dir; repeatable; 'none' = base model (default)")
    p.add_argument("--k", type=int, default=16, help="samples per prompt at --temperature (0 = skip)")
    p.add_argument("--no-greedy", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--completion-tokens", type=int, default=DEFAULT_COMPLETION_TOKENS)
    p.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
    p.add_argument("--seed", type=int, default=0, help="generation seed")
    p.add_argument("--sample-seed", type=int, default=0, help="seed for :limit prompt subsets")
    p.add_argument("--model-id", default=None, help="default: grpo.grpo.MODEL_ID")
    p.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--batch-size", type=int, default=8, help="hf backend prompts per batch")
    p.add_argument("--score-workers", type=int, default=8, help="parallel scoring (kodcode / opencodeinstruct only)")
    p.add_argument("--save-completions", choices=("none", "greedy", "all"), default="greedy")
    p.add_argument("--output-dir", default=None, help="only valid with a single adapter/base")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    specs = parse_split_specs(args.splits)
    adapters = [None if a in (None, "none") else a for a in (args.adapter or ["none"])]
    adapters.sort(key=lambda a: a is not None)   # base first: no weight push needed
    if args.output_dir and len(adapters) > 1:
        raise SystemExit("--output-dir only works with one --adapter")
    if args.k < 0 or (args.k == 0 and args.no_greedy):
        raise SystemExit("nothing to do: --k 0 with --no-greedy")

    model_id = args.model_id
    if model_id is None:
        import grpo.grpo as g
        model_id = g.MODEL_ID
    infos = {a: adapter_info(a) for a in adapters}   # fail fast on bad paths

    # Plan outputs before loading any model.
    plan = []
    for a in adapters:
        out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(model_id, args.task, a)
        for split, limit in specs:
            stem = f"{split}_n{limit}" if limit else split
            if (out_dir / f"{stem}.json").exists() and not args.overwrite:
                raise SystemExit(f"{out_dir / stem}.json exists (pass --overwrite)")
            plan.append((a, out_dir, split, limit, stem))

    examples_by_split: dict[tuple[str, int | None], tuple[list[Example], dict]] = {}
    for split, limit in specs:
        exs = load_examples(args.dataset, split, args.task)
        idx = sample_indices(len(exs), limit, args.sample_seed)
        f = resolve_split_file(args.dataset, split)
        src = {"dataset": args.dataset, "split": split, "file": str(f) if f else None,
               "file_sha256": sha256_file(f) if f else None, "n_rows": len(exs),
               "limit": limit, "sample_seed": args.sample_seed, "n_eval": len(idx)}
        examples_by_split[(split, limit)] = ([exs[i] for i in idx], src)
        print(f"[eval] {split}: {len(idx)}/{len(exs)} prompts")

    if args.backend == "vllm":
        backend = VLLMBackend(model_id, args.context_tokens, args.seed, args.gpu_memory_utilization)
    else:
        backend = HFBackend(model_id, args.context_tokens, args.seed, args.batch_size)
    scorer = Scorer(args.task, args.score_workers)

    for a, out_dir, split, limit, stem in plan:
        backend.load_adapter(a)
        examples, src = examples_by_split[(split, limit)]
        started = _dt.datetime.now(_dt.timezone.utc)
        records, completions = evaluate_split(
            backend, scorer, examples, k=args.k, greedy=not args.no_greedy,
            temperature=args.temperature, max_tokens=args.completion_tokens, seed=args.seed,
            save_completions=args.save_completions)
        summary = {
            "task": args.task, "model_id": model_id, **infos[a], **src,
            "metrics": summarize(records),
            "generation": {"backend": backend.name, "k": args.k, "greedy": not args.no_greedy,
                           "temperature": args.temperature, "top_p": 1.0,
                           "max_completion_tokens": args.completion_tokens,
                           "context_tokens": args.context_tokens, "seed": args.seed},
            "scorer": "env.step() via grpo env classes; unscored -> 0 (make_env_reward_fn)",
            "buckets": {"hard": f"p<{HARD_BELOW}", "easy": f"p>{EASY_ABOVE}"},
            "git_sha": _git_sha(), "versions": _versions(),
            "started_utc": started.isoformat(),
            "seconds": (_dt.datetime.now(_dt.timezone.utc) - started).total_seconds(),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{stem}_per_prompt.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        if completions:
            with open(out_dir / f"{stem}_completions.jsonl", "w") as f:
                for c in completions:
                    f.write(json.dumps(c) + "\n")
        (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
        m = summary["metrics"]
        line = f"[eval] {a or 'base'} {split}{'' if not limit else f':{limit}'}"
        if "greedy" in m:
            line += f" | greedy acc {m['greedy']['accuracy']:.3f}"
        if "sampled" in m:
            s = m["sampled"]
            line += (f" | pass@1 {s['pass_at']['1']:.3f} | buckets "
                     + " ".join(f"{b}={v:.2f}" for b, v in s["buckets"].items()))
        print(line + f" -> {out_dir / stem}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
