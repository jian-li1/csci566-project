# GRPO (shared trainer)

One TRL `GRPOTrainer` script for Reasoning Gym tasks, Competition Math, and a coding stub. Rewards come from task **environments**; logging and analysis go through **metrics**; compute-budget stopping uses a **team FLOP-weighted token model** (validate on a smoke run before scientific experiments).

Entry point: [`grpo.py`](grpo.py).

## How a step works

```
dataset row
  → normalize → {prompt, task, entry}
  → model samples G completions per prompt
  → make_env_reward_fn (metrics.py)
        env.load_problem(entry)
        env.step(completion) → reward
        metrics.record(...) → rollouts.jsonl (+ later W&B via callback)
  → GRPO group-normalize + LoRA update
  → callbacks: completion tokens, compute budget, metrics flush
```

**One run = one `--task`.** The dataset’s `task` column is set to that name for every row so the metrics reward can pick the right env.

### Ablation geometry (single-GPU assumption)

| Knob | Value |
|------|--------|
| Completions / optimizer step | 64 (**per device**; multi-GPU changes global effective batch) |
| `--G` (completions per prompt) | 2, 4, 8, or 16 |
| Prompts / step | `64 / G` |
| Stop | `--max-steps` and/or `--budget` (after token accounting is verified) |

Equal optimizer steps alone do **not** imply equal FLOPs across setups; use measured weighted tokens from `compute.json`.

## Environments

Each env owns prompting, verification, and the `step()` reward. Shared contract for the metrics reward:

```python
env.load_problem(entry)   # entry is a dict
out = env.step(text)      # out["reward"], out["acceptor"], out["check"], ...
```

| `--task` | Env | `entry` fields |
|----------|-----|----------------|
| `countdown` | `CountdownEnv` (RG) | `question`, `answer`, optional `metadata` |
| `knights_knaves` | `KnightsKnavesEnv` (RG) | same |
| `shortest_path` | `ShortestPathEnv` (RG) | same |
| `math` | `CompetitionMathEnv` | `problem`, `solution`, optional `level`, `type` |
| `coding` | *(stub)* | pass-through; **zero reward** until MBPP is wired |

- **Reasoning Gym wrappers** — thin adapters over [`reasoning_gym_env.py`](reasoning_gym_env.py).
- **Math** — [`math/environment.py`](math/environment.py). Supports `load_problem(entry: dict)` (metrics path) and the legacy positional API.
- **MBPP** — [`mbpp/`](mbpp/) has its own env + `train.py`; not yet hooked into shared `grpo.py` as `--task coding`.

Normalize ([`_normalize_row`](grpo.py)) builds `prompt` via the env and stores `task` + `entry` for the reward.

## Metrics ([`metrics.py`](metrics.py))

Owner: Manasa. Hookup inside `grpo.py`:

```python
metrics = GRPOMetrics(output_dir, max_completion_length=1024)
reward_fn = make_env_reward_fn({args.task: make_env_for_task(args.task)}, metrics)
# callbacks=[token_cb, budget_cb, GRPOMetricsCallback(metrics)]
```

`make_env_reward_fn` is the **single** TRL reward path for RG/math. It scores with the env and records every rollout into a buffer / `rollouts.jsonl`. Derived scalars are merged into Trainer logs by `GRPOMetricsCallback` (W&B picks those up if enabled)—recording alone does not guarantee every metric appears in W&B.

Logged (per step / per task), among other things:

- reward mean/std/min/max, accept rate  
- failure rates (`missing_answer`, `empty_answer`, `verifier_crash`, …)  
- fraction of groups with zero reward std (no GRPO signal)  
- advantage stats  
- length / length-limit proxy (`len/.../hit_length_limit_rate` aka `trunc_rate`: `n_tokens >= max_completion_length`; needs `max_completion_length` on `GRPOMetrics` **and** TRL `completion_ids`. This is not confirmed EOS-absence truncation.)  
- alerts: KL spike, entropy collapse, reward-std collapse  

Artifacts:

- `<output_dir>/rollouts.jsonl` — every completion for offline failure analysis  
- W&B metrics under `reward/`, `fail/`, `group/`, `adv/`, `alert/`, … (when `report_to` includes wandb)

Offline summary:

```bash
python -m grpo.metrics <output_dir>
```

## Compute budget ([`../compute_budget/`](../compute_budget/))

Shared with SFT. See [`../compute_budget/README.md`](../compute_budget/README.md).

**Intent:** stop when weighted tokens hit `B`. **Before scientific runs**, verify on a short job that:

1. `state.num_input_tokens_seen` includes the intended prompt + completion tokens under TRL GRPO  
2. the counter increases each optimizer step (not stuck at 0)  
3. `--budget` actually stops training when reached  

The callback reads `num_input_tokens_seen` deltas. If the counter never moves, training continues and only a warning is printed.

| CLI | Meaning |
|-----|---------|
| `--budget B` | Total weighted tokens (FLOPs / N) for this model. Omit to track without stopping. |
| `--stage none` | Use 100% of `B` (pure GRPO). Default. |
| `--stage sft_then_grpo:grpo` | Use 75% of `B` (GRPO half of SFT→GRPO). |

This script does **not** accept `sft_then_grpo:sft` (that stage belongs to the SFT trainer).

With LoRA, μ=1, β>0, weight is **8×** per GRPO rollout token. Writes `<output_dir>/compute.json`.

Also kept: `CompletionTokenCallback` → `completion_tokens.json` (approx. completion tokens for accuracy-vs-tokens plots).

## Dataset

**Local file:** `.parquet` / `.jsonl` / `.json`  
**Directory:** looks for `<split>.parquet`, then `<split>.jsonl`, then `<split>.json` (`--dataset-split`, default `train`)  
**Hub:** Hugging Face dataset id + `--dataset-split`

Raw rows (before normalize):

| Task | Expected columns |
|------|------------------|
| RG | `question`, `answer`, optional `metadata` |
| Math | `problem`, `solution`, optional `level`, `type` |

After normalize, required columns are always: **`prompt`**, **`task`**, **`entry`**.

Math drops rows whose solution has no usable `\boxed{...}` gold before normalize.

## How to run

Prefer module launch from the **repo root** (avoids putting `grpo/` first on `sys.path`, which can shadow the stdlib `math` module via `grpo/math/`):

```bash
cd /path/to/csci566-project
export PYTHONPATH="$PWD"
export WANDB_MODE=offline   # optional for smoke

python -m grpo.grpo \
  --task countdown \
  --G 4 \
  --dataset /path/to/countdown/train.jsonl \
  --max-steps 2 \
  --output-dir runs/smoke-countdown
```

`grpo.py` also inserts the repo root into `sys.path` before local imports, so `python grpo/grpo.py ...` often works, but `-m` is preferred.

vLLM colocate is **on by default**. For a first smoke, consider temporarily setting `use_vllm=False` in `GRPOConfig` (no CLI flag yet) to reduce moving parts; re-enable and validate the Qwen3.5 weight-sync patch afterward.

### Smoke-test success criteria

A 2-step Countdown (or Math) smoke should show:

- [ ] At least two optimizer steps complete without exceptions  
- [ ] Environment rewards are finite (not all NaN); ideally some variation across the group  
- [ ] Training loss is finite  
- [ ] `rollouts.jsonl` contains real generated completions  
- [ ] `compute.json` / budget logs show **increasing** token counts (not stuck at 0)  
- [ ] A LoRA adapter is written under `--output-dir` and loads for a quick generate  

**Checkpoint / resume** is a **separate** test: with `save_steps=50`, a 2-step run only hits the final `save_model`, not mid-run optimizer-state recovery. Use a longer run or lower `save_steps` to exercise resume.

## CLI cheat sheet

| Flag | Role |
|------|------|
| `--task` | `countdown` \| `knights_knaves` \| `shortest_path` \| `math` \| `coding` |
| `--G` | 2 \| 4 \| 8 \| 16 |
| `--dataset` | path or HF id |
| `--dataset-split` | default `train` |
| `--lr`, `--max-steps`, `--seed` | training |
| `--init-adapter` | continue from SFT LoRA |
| `--output-dir` | default `GRPO/grpo-{task}-G{G}-…` |
| `--budget`, `--stage` | compute budget (`none` \| `sft_then_grpo:grpo`) |

## Preflight checklist

- [ ] Repo at known git commit; `PYTHONPATH` = repo root  
- [ ] CUDA GPU job (CARC); Mac is for env unit tests only  
- [ ] Pinned Transformers / TRL / PEFT / (vLLM) that support `MODEL_ID` (e.g. Qwen3.5)  
- [ ] Task deps installed (`requirements-reasoning-gym.txt` and/or `math/requirements.txt`)  
- [ ] `python -m unittest grpo.math.test_environment` (or RG env smoke) passes  
- [ ] Dataset path/schema verified; normalize produces `prompt` / `task` / `entry`  
- [ ] Smoke criteria above checked before long runs  

## Experiment reproducibility (log these)

Exact git commit, dataset path + hash/manifest, `MODEL_ID` revision, `--seed`, `--G`, batch geometry (64 / microbatch / grad accum), LR, `max_completion_length`, generation settings (temp, β), measured weighted tokens (`compute.json`), GPU type/count, whether vLLM was enabled.

## Dependencies

- TRL / Transformers / PEFT / datasets / vLLM (pin for your `MODEL_ID`)  
- Reasoning Gym: [`requirements-reasoning-gym.txt`](requirements-reasoning-gym.txt)  
- Math: [`math/requirements.txt`](math/requirements.txt)  
- Compute budget: [`../compute_budget/`](../compute_budget/)  

`MODEL_ID` is a constant at the top of [`grpo.py`](grpo.py).

## Status / caveats

- **`--task coding`** — no-op reward; do not use for real training until MBPP is wired.  
- **Budget / FLOP matching** — designed in; **verify token counters on CARC** before trusting matched-compute claims.  
- **Per-model `B`** — still TBD by the team.  
- Standalone trainers remain under [`math/train.py`](math/train.py) and [`mbpp/train.py`](mbpp/train.py); shared experiments should prefer `grpo.py`.
