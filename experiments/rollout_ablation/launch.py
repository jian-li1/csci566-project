"""
Launch and track the GRPO group-size ablation on CARC (owner: Mehul).

    # what would run, with all pre-flight checks (no submission)
    python -m experiments.rollout_ablation.launch plan   --task shortest_path --dataset data/shortest_path --stage 1
    # submit (one SLURM job per arm; each resubmits itself after time-limit stops)
    python -m experiments.rollout_ablation.launch submit --task shortest_path --dataset data/shortest_path --stage 1
    # progress table: steps, attempts, validation, evals, init-adapter/data hashes
    python -m experiments.rollout_ablation.launch status --task shortest_path
    # restart unfinished arms (after a crash or MAX_ATTEMPTS)
    python -m experiments.rollout_ablation.launch submit --task shortest_path --dataset data/shortest_path --stage 1 --resume

Arms are G in config.G_VALUES x the stage's seeds (or --G / --seeds). Every arm gets
the same dataset file (sha256 checked), init adapter, hyperparameters and step
count; manifest.json in each run dir records them and a later --resume must match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rollout_ablation import config as C  # noqa: E402

TRAIN_SBATCH = C.REPO_ROOT / "experiments" / "rollout_ablation" / "train.sbatch"
MANIFEST_KEYS = ("task", "G", "seed", "dataset_file", "dataset_sha256", "hparams",
                 "init_adapter")   # must match on --resume


# ---------------------------------------------------------------------------
# Dataset checks
# ---------------------------------------------------------------------------
def resolve_split_file(dataset: str, split: str) -> Path | None:
    p = Path(dataset)
    if p.is_file():
        return p if split == "train" else None
    if p.is_dir():
        for ext in ("parquet", "jsonl", "json"):
            if (p / f"{split}.{ext}").is_file():
                return p / f"{split}.{ext}"
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path: Path) -> int | None:
    if path.suffix == ".jsonl":
        with open(path, "rb") as f:
            return sum(1 for line in f if line.strip())
    if path.suffix == ".json":
        try:
            d = json.loads(path.read_text())
            return len(d) if isinstance(d, list) else None
        except Exception:
            return None
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def prompts_needed(G: int, max_steps: int) -> int:
    return max_steps * (C.ROLLOUTS_PER_STEP // G)


def effective_hparams(epochs: float | None = None, max_steps: int | None = None,
                      completion_tokens: int | None = None) -> dict[str, Any]:
    """config.HPARAMS with CLI overrides; exactly one of epochs / max_steps is set."""
    hp = dict(C.HPARAMS)
    if max_steps is not None:
        hp.update(max_steps=max_steps, epochs=None)
    elif epochs is not None:
        hp.update(epochs=epochs, max_steps=None)
    if completion_tokens is not None:
        hp["completion_tokens"] = completion_tokens
    return hp


def dataset_checks(dataset: str, Gs: list[int], hp: dict[str, Any] | int, *, allow_repeats: bool,
                   need_eval: bool) -> tuple[dict[str, Any], list[str], list[str]]:
    """Returns (info, errors, warnings). `hp` is effective_hparams() (or a bare max_steps)."""
    if not isinstance(hp, dict):
        hp = effective_hparams(max_steps=hp)
    max_steps = hp["max_steps"]
    errors, warnings = [], []
    info: dict[str, Any] = {"dataset": dataset}
    train = resolve_split_file(dataset, "train")
    if train is None:
        return info, [f"no train split found under {dataset}"], warnings
    info.update(dataset_file=str(train.resolve()), dataset_sha256=sha256_file(train))
    n = count_rows(train)
    info["n_rows"] = n
    if n is None:
        warnings.append(f"could not count rows in {train}; prompt-coverage check skipped")
    else:
        for G in Gs:
            info[f"steps_G{G}"] = C.steps_for(G, n, hp["epochs"], max_steps)
            if not max_steps:
                continue     # epochs rule: every prompt is seen `epochs` times in every arm
            need = prompts_needed(G, max_steps)
            info[f"prompts_needed_G{G}"] = need
            if n < need:
                msg = (f"G={G}: {max_steps} steps x {C.ROLLOUTS_PER_STEP // G} prompts = {need} "
                       f"prompts but train has {n} rows -> prompts repeat (epoch {need / n:.2f})")
                (warnings if allow_repeats else errors).append(msg)
    val = resolve_split_file(dataset, C.EVAL["split"])
    if val is None:
        (errors if need_eval else warnings).append(
            f"no {C.EVAL['split']} split under {dataset} for checkpoint evals (or pass --no-eval)")
    else:
        info.update(eval_file=str(val.resolve()), eval_sha256=sha256_file(val))
    return info, errors, warnings


# ---------------------------------------------------------------------------
# Arms, manifests, status
# ---------------------------------------------------------------------------
def arms(Gs: list[int], seeds: list[int]) -> list[tuple[int, int]]:
    return [(G, s) for s in seeds for G in Gs]


def build_manifest(task: str, G: int, seed: int, data: dict[str, Any],
                   hp: dict[str, Any] | None = None) -> dict[str, Any]:
    hp = hp or effective_hparams()
    return {
        "task": task, "G": G, "seed": seed,
        "prompts_per_step": C.ROLLOUTS_PER_STEP // G, "rollouts_per_step": C.ROLLOUTS_PER_STEP,
        "dataset": data["dataset"], "dataset_file": data["dataset_file"],
        "dataset_sha256": data["dataset_sha256"], "n_rows": data.get("n_rows"),
        "eval_file": data.get("eval_file"), "eval_sha256": data.get("eval_sha256"),
        "hparams": dict(hp), "planned_steps": data.get(f"steps_G{G}"),
        "init_adapter": str(C.INIT_ADAPTER_DIR),
        "init_adapter_seed": C.INIT_ADAPTER_SEED, "eval": dict(C.EVAL),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def manifest_mismatch(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    return [f"{k}: {old.get(k)!r} -> {new.get(k)!r}" for k in MANIFEST_KEYS if old.get(k) != new.get(k)]


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def run_status(rd: Path) -> dict[str, Any]:
    st: dict[str, Any] = {"exists": rd.is_dir()}
    if not rd.is_dir():
        return st
    s = _read_json(rd / "status.json") or {}
    steps = [int(p.name.split("-")[1]) for p in rd.glob("checkpoint-*")
             if p.name.split("-")[1].isdigit()]
    attempts = _read_jsonl(rd / "attempts.jsonl")
    jobs = [j for j in _read_jsonl(rd / "jobs.jsonl") if j.get("slurm_job_id")]
    val = _read_json(rd / "rollout_validation_offline.json")
    evals = sorted(rd.glob(f"checkpoint-*/eval/*/{C.EVAL['splits'].replace(':', '_n')}.json"))
    st.update(
        done=bool(s.get("done")), global_step=s.get("global_step", max(steps, default=0)),
        max_steps=s.get("max_steps"),
        checkpoints=sorted(steps), attempts=len(attempts),
        last_job=jobs[-1]["slurm_job_id"] if jobs else None,
        validation=None if val is None else ("PASS" if val.get("passed") else "FAIL"),
        n_evals=len(evals),
        init_sha={a.get("init_adapter_sha256") for a in attempts if a.get("init_adapter_sha256")},
        git_shas={a.get("git_sha") for a in attempts if a.get("git_sha")},
        dirty=any(a.get("git_dirty") for a in attempts),
    )
    return st


def squeue_states(job_ids: list[str]) -> dict[str, str]:
    if not job_ids or shutil.which("squeue") is None:
        return {}
    try:
        out = subprocess.check_output(["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i %T"],
                                      text=True, stderr=subprocess.DEVNULL)
        return dict(line.split() for line in out.splitlines() if line.strip())
    except Exception:
        return {}


def print_status(task: str, Gs: list[int] | None, seeds: list[int] | None) -> int:
    root = C.RUN_ROOT / task
    found = []
    for rd in sorted(root.glob("G*-s*")):
        m = re.fullmatch(r"G(\d+)(?:-L\d+)?-s(\d+)", rd.name)
        if not m:
            continue
        G, s = int(m.group(1)), int(m.group(2))
        if (Gs and G not in Gs) or (seeds and s not in seeds):
            continue
        found.append((G, s, rd, run_status(rd)))
    if not found:
        print(f"no runs under {root}")
        return 1
    states = squeue_states([st["last_job"] for *_, st in found if st.get("last_job")])
    print(f"{'arm':<16}{'step':>9}  {'state':<12}{'att':>4}  {'valid':<6}{'evals':>6}  init     git")
    init_all, git_all = set(), set()
    for G, s, rd, st in sorted(found):
        state = "done" if st["done"] else states.get(st.get("last_job") or "", "stopped")
        init_all |= st["init_sha"]
        git_all |= st["git_shas"]
        init = ",".join(x[:7] for x in st["init_sha"]) or "-"
        git = ",".join(x[:7] for x in st["git_shas"]) + ("*" if st["dirty"] else "")
        total = st.get("max_steps") or (_read_json(rd / "manifest.json") or {}).get("planned_steps") or "?"
        print(f"{rd.name:<16}{st['global_step']:>5}/{total!s:<4} {state:<12}{st['attempts']:>4}  "
              f"{st['validation'] or '-':<6}{st['n_evals']:>6}  {init:<8} {git}")
    if len(init_all) > 1:
        print("WARNING: arms started from DIFFERENT init adapters:", init_all)
    if len(git_all) > 1:
        print("NOTE: arms ran on different git commits:", {x[:7] for x in git_all})
    return 0


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
def job_env(task: str, G: int, seed: int, rd: Path, dataset: str, args,
            hp: dict[str, Any] | None = None) -> dict[str, str]:
    h = hp or effective_hparams()
    stop = {"MAX_STEPS": str(h["max_steps"])} if h["max_steps"] else {"EPOCHS": str(h["epochs"])}
    return {
        **stop,
        "RUN_DIR": str(rd), "TASK": task, "G": str(G), "SEED": str(seed),
        "DATASET": str(Path(dataset).resolve()), "INIT_ADAPTER": str(C.INIT_ADAPTER_DIR),
        "LR": str(h["lr"]),
        "COMPLETION_TOKENS": str(h["completion_tokens"]), "CONTEXT_TOKENS": str(h["context_tokens"]),
        "ATTEMPT": "1", "MAX_ATTEMPTS": str(C.MAX_ATTEMPTS), "JOB_TIME": args.time,
        "STOP_MARGIN": str(C.STOP_MARGIN_SECONDS),
        "EVAL_SPLITS": C.EVAL["splits"], "EVAL_K": str(C.EVAL["k"]),
        "NO_EVAL": "1" if args.no_eval else "0",
    }


def submit(args, Gs: list[int], seeds: list[int]) -> int:
    hp = effective_hparams(args.epochs, args.max_steps, args.completion_tokens)
    data, errors, warnings = dataset_checks(args.dataset, Gs, hp,
                                            allow_repeats=args.allow_repeats,
                                            need_eval=not args.no_eval)
    print(f"task={args.task} dataset={data.get('dataset_file')} rows={data.get('n_rows')} "
          f"sha256={str(data.get('dataset_sha256'))[:12]}")
    print(f"hparams={hp} init_adapter={C.INIT_ADAPTER_DIR}")
    print("steps per arm: " + ", ".join(f"G={G}: {data.get(f'steps_G{G}')}" for G in Gs)
          + ("" if hp["max_steps"] else f"  ({hp['epochs']:g} epochs over {data.get('n_rows')} prompts)"))
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)
    if errors:
        return 2
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=C.REPO_ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        print("WARNING: tracked files have uncommitted changes; jobs run the working tree "
              "as it is when each job starts")

    rows = []
    for G, seed in arms(Gs, seeds):
        rd = C.run_dir(args.task, G, seed, hp["completion_tokens"])
        new = build_manifest(args.task, G, seed, data, hp)
        st = run_status(rd)
        action = "submit"
        if st["exists"] and st.get("done"):
            action = "skip (done)"
        elif st["exists"]:
            old = _read_json(rd / "manifest.json") or {}
            bad = manifest_mismatch(old, new)
            if bad:
                action = "REFUSE (manifest differs: " + "; ".join(bad) + ")"
            elif not args.resume:
                action = f"skip (exists at step {st['global_step']}; --resume to continue)"
            elif st.get("last_job") and squeue_states([st["last_job"]]):
                action = f"skip (job {st['last_job']} still queued/running)"
            else:
                action = "resume"
        rows.append((G, seed, rd, new, action))

    for G, seed, rd, _, action in rows:
        print(f"  G{G:<3} seed {seed:<4} {action:<50} {rd}")
    if args.command == "plan" or args.dry_run:
        print("(plan only; nothing submitted)" if args.command == "plan" else "(dry run)")
        if args.dry_run:
            for G, seed, rd, _, action in rows:
                if action in ("submit", "resume"):
                    env = job_env(args.task, G, seed, rd, args.dataset, args, hp)
                    print("   ", " ".join(f"{k}={v}" for k, v in env.items()),
                          "sbatch", f"--job-name=abl-{args.task}-G{G}-s{seed}",
                          f"--time={args.time}", str(TRAIN_SBATCH.relative_to(C.REPO_ROOT)))
        return 0

    if shutil.which("sbatch") is None:
        print("ERROR: sbatch not found (run on a CARC login node)")
        return 2
    log = C.RUN_ROOT / args.task / "launches.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    (C.REPO_ROOT / "logs").mkdir(exist_ok=True)
    for G, seed, rd, manifest, action in rows:
        if action not in ("submit", "resume"):
            continue
        rd.mkdir(parents=True, exist_ok=True)
        if not (rd / "manifest.json").is_file():
            (rd / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        env = {**os.environ, **job_env(args.task, G, seed, rd, args.dataset, args, hp)}
        env.pop("MAX_STEPS" if "EPOCHS" in env else "EPOCHS", None)   # never both
        out = subprocess.run(["sbatch", "--parsable", f"--job-name=abl-{args.task}-G{G}-s{seed}",
                              f"--time={args.time}", "--export=ALL", str(TRAIN_SBATCH)],
                             cwd=C.REPO_ROOT, env=env, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"  G{G} seed {seed}: sbatch failed: {out.stderr.strip()}")
            continue
        job = out.stdout.strip().split(";")[0]
        print(f"  G{G} seed {seed}: job {job}")
        with open(log, "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "G": G, "seed": seed,
                                "job": job, "action": action, "run_dir": str(rd)}) + "\n")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("plan", "submit", "status"))
    p.add_argument("--task", required=True,
                   choices=("countdown", "knights_knaves", "shortest_path", "math", "kodcode",
                            "opencodeinstruct"))
    p.add_argument("--dataset", help="dir with train.* (+ val.* for checkpoint evals)")
    p.add_argument("--stage", type=int, choices=sorted(C.STAGE_SEEDS), default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="overrides --stage")
    p.add_argument("--G", type=int, nargs="+", default=None, choices=C.G_VALUES)
    stop = p.add_mutually_exclusive_group()
    stop.add_argument("--epochs", type=float, default=None,
                      help=f"passes over the train file (default {C.HPARAMS['epochs']:g})")
    stop.add_argument("--max-steps", type=int, default=None,
                      help="equal-steps rule instead of epochs")
    p.add_argument("--completion-tokens", type=int, default=None,
                   help=f"completion cap (default {C.HPARAMS['completion_tokens']}); a different "
                        "value becomes its own arm directory, e.g. G4-L2048-s42")
    p.add_argument("--resume", action="store_true", help="continue existing unfinished arms")
    p.add_argument("--allow-repeats", action="store_true",
                   help="allow fewer train rows than the G=4 arm consumes (prompts repeat)")
    p.add_argument("--no-eval", action="store_true", help="do not queue checkpoint evals")
    p.add_argument("--time", default=C.SLURM_TIME)
    p.add_argument("--dry-run", action="store_true", help="print sbatch commands, submit nothing")
    a = p.parse_args(argv)

    Gs = sorted(a.G or C.G_VALUES)
    seeds = a.seeds or (list(C.STAGE_SEEDS[a.stage]) if a.stage else None)
    if a.command == "status":
        return print_status(a.task, a.G, seeds)
    if not a.dataset or not seeds:
        p.error("plan/submit need --dataset and --stage (or --seeds)")
    return submit(a, Gs, seeds)


if __name__ == "__main__":
    sys.exit(main())
