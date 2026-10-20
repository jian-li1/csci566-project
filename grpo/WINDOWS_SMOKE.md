# Windows laptop smoke tests

Two layers:

1. **CPU (everyone)** — Math + RG envs and metrics reward plumbing (sections 2–4).  
2. **Light GPU (optional)** — 1–2 GRPO steps **without vLLM** if the laptop has an **NVIDIA GPU with enough VRAM** (section 5). ~24 GB **system RAM** alone is not enough; you need GPU memory (roughly **≥12 GB VRAM** for Qwen3.5-4B + LoRA is a practical minimum; 16–24 GB VRAM is more comfortable).

| Check | CPU | Light GPU (`--no-vllm`) | Full CARC (vLLM) |
|-------|-----|-------------------------|------------------|
| Math / RG env + metrics hook | yes | yes | yes |
| 1–2 GRPO optimizer steps | no | yes (if VRAM OK) | yes |
| Compute-budget / scientific FLOPs | no | smoke only | yes |
| Colocated vLLM + Qwen sync | no | no (skip) | yes |

**CPU pass definition:** unit tests + reward smokes (sections 2–4).  
**Light GPU pass definition:** those plus a 1–2 step Countdown GRPO run that finishes, writes `rollouts.jsonl`, and saves a LoRA adapter. Still **not** a matched-compute CARC experiment.

Tests live on **`main`** (pull latest before running).

---

## 1. Setup (PowerShell)

Avoids venv “Activate.ps1” / execution-policy issues by calling the venv Python directly.

```powershell
git clone https://github.com/jian-li1/csci566-project.git
cd csci566-project
git switch main
git pull
git rev-parse --short HEAD

# Prefer 3.10 or 3.11 if installed via the Python launcher:
py -3.11 -m venv .venv
# If that fails:  python -m venv .venv

$py = ".\.venv\Scripts\python.exe"

& $py -m pip install --upgrade pip
& $py -m pip install "math-verify[antlr4_13_2]"
& $py -m pip install -r grpo\requirements-reasoning-gym.txt

$env:PYTHONPATH = (Get-Location).Path
```

Report these with your results:

```powershell
git rev-parse --short HEAD
& $py --version
& $py -m pip show math-verify reasoning-gym | Select-String "Name|Version"
```

---

## 2. Syntax + Math unit tests (required)

```powershell
& $py -m compileall -q grpo
& $py -m unittest grpo.math.test_environment -v
```

**Expect:** all tests `ok`, including:

- `test_load_problem_accepts_metrics_entry_dict`
- `test_metrics_reward_hook_loads_entry_dict`
- correct / wrong / missing-`<answer>` cases

---

## 3. Math metrics reward smoke (required)

Correct answer → reward `1.0` + `rollouts.jsonl`. Unexpected `None` is a **failure**.

```powershell
$code = @'
from pathlib import Path
from tempfile import TemporaryDirectory
from grpo.math import CompetitionMathEnv
from grpo.metrics import GRPOMetrics, make_env_reward_fn

entry = {
    "problem": "Compute 1+1.",
    "solution": r"Clearly \boxed{2}.",
}

with TemporaryDirectory() as tmp:
    fn = make_env_reward_fn(
        {"math": CompetitionMathEnv()},
        GRPOMetrics(tmp),
    )
    rewards = fn(
        prompts=["Compute 1+1."],
        completions=["<answer>2</answer>"],
        task=["math"],
        entry=[entry],
    )
    print("Rewards:", rewards)
    assert len(rewards) == 1, rewards
    assert rewards[0] == 1.0, rewards
    assert (Path(tmp) / "rollouts.jsonl").exists()

print("MATH REWARD SMOKE PASSED")
'@

& $py -c $code
```

Wrong / malformed (should **not** be unscored `None` for these cases — expect `0.0`):

```powershell
$code = @'
from tempfile import TemporaryDirectory
from grpo.math import CompetitionMathEnv
from grpo.metrics import GRPOMetrics, make_env_reward_fn

entry = {
    "problem": "Compute 1+1.",
    "solution": r"Clearly \boxed{2}.",
}

with TemporaryDirectory() as tmp:
    fn = make_env_reward_fn({"math": CompetitionMathEnv()}, GRPOMetrics(tmp))
    wrong = fn(
        prompts=["p"], completions=["<answer>3</answer>"],
        task=["math"], entry=[entry],
    )
    missing = fn(
        prompts=["p"], completions=["The answer is 2."],
        task=["math"], entry=[entry],
    )
    print("wrong", wrong, "missing_tags", missing)
    assert wrong == [0.0], wrong
    assert missing == [0.0], missing

print("MATH NEGATIVE CASES PASSED")
'@

& $py -c $code
```

---

## 4. Reasoning Gym envs (required) — countdown, knights_knaves, shortest_path

Each env: `reset` → metrics reward with gold `<answer>` → reward finite and typically accepted; malformed completion scored (not a crash).

```powershell
$code = @'
from tempfile import TemporaryDirectory
from grpo.countdown import CountdownEnv
from grpo.knights_knaves import KnightsKnavesEnv
from grpo.shortest_path import ShortestPathEnv
from grpo.metrics import GRPOMetrics, make_env_reward_fn

TASKS = {
    "countdown": CountdownEnv,
    "knights_knaves": KnightsKnavesEnv,
    "shortest_path": ShortestPathEnv,
}

for name, Env in TASKS.items():
    print(f"=== {name} ===")
    env = Env()
    obs = env.reset(index=0)
    assert "prompt" in obs and obs["prompt"], obs
    entry = env.entry
    assert isinstance(entry, dict) and "question" in entry and "answer" in entry, entry
    gold = str(entry["answer"])

    with TemporaryDirectory() as tmp:
        fn = make_env_reward_fn({name: Env()}, GRPOMetrics(tmp))
        # Correct-format gold (RG may still give partial credit; must be a float, not None)
        ok = fn(
            prompts=[obs["prompt"]],
            completions=[f"<answer>{gold}</answer>"],
            task=[name],
            entry=[entry],
        )
        bad = fn(
            prompts=[obs["prompt"]],
            completions=["no answer tags"],
            task=[name],
            entry=[entry],
        )
        print("  gold_reward", ok, "malformed_reward", bad)
        assert len(ok) == 1 and ok[0] is not None, ok
        assert isinstance(ok[0], float), ok
        assert len(bad) == 1 and bad[0] is not None, bad
        assert bad[0] == 0.0, bad

print("ALL RG ENV SMOKES PASSED")
'@

& $py -c $code
```

**Notes**

- First `Env()` may download / build the RG dataset (needs network once).
- Gold reward is often `1.0` but RG can use partial credit; we only require a **non-`None` float**. Malformed (no tags) must be **`0.0`**.
- If a task fails, report which name + full traceback.

---

## 5. Optional — light GRPO on a Windows NVIDIA GPU

Skip this if there is **no NVIDIA GPU**, or if `nvidia-smi` shows little free VRAM. **Do not use vLLM on Windows** for this smoke (`--no-vllm`).

### 5a. Check the GPU

```powershell
nvidia-smi
```

You want a CUDA GPU with enough **VRAM** (memory column), not just 24 GB of system RAM.

### 5b. Extra packages (still no vLLM)

Install a **CUDA** PyTorch build from [pytorch.org](https://pytorch.org) (pick your CUDA version), then:

```powershell
& $py -m pip install transformers datasets peft trl accelerate wandb
# Do NOT install trl[vllm] / vllm for this Windows smoke.
```

Confirm CUDA:

```powershell
& $py -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

### 5c. Tiny Countdown dataset

```powershell
New-Item -ItemType Directory -Force -Path data\smoke | Out-Null

$code = @'
import json
from pathlib import Path
from grpo.countdown import CountdownEnv

env = CountdownEnv()
path = Path("data/smoke/train.jsonl")
with path.open("w", encoding="utf-8") as f:
    for i in range(32):
        env.reset(index=i)
        e = env.entry
        row = {"question": e["question"], "answer": e["answer"]}
        if e.get("metadata") is not None:
            row["metadata"] = e["metadata"]
        f.write(json.dumps(row) + "\n")
print("wrote", path, "rows", 32)
'@

& $py -c $code
```

### 5d. One-step (or two-step) GRPO smoke

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:WANDB_MODE = "offline"

& $py -m grpo.grpo `
  --task countdown `
  --G 4 `
  --dataset data\smoke\train.jsonl `
  --max-steps 1 `
  --no-vllm `
  --report-to none `
  --output-dir runs\win-gpu-smoke
```

Optional second step: `--max-steps 2`.

**Success**

- [ ] Run exits 0 (no traceback / CUDA OOM)  
- [ ] Logs show a finite training loss  
- [ ] `runs\win-gpu-smoke\rollouts.jsonl` exists and has completions  
- [ ] Adapter / model files appear under `runs\win-gpu-smoke`  
- [ ] (Bonus) `compute.json` exists; token counts are not stuck at 0  

**If you OOM:** stop here and leave full training for CARC; CPU sections 2–4 still count as a useful env/metrics cert.

Full CARC / vLLM runs: see [`README.md`](README.md).

---

## 6. Report back (copy/paste)

```
git commit:     ________
python:         ________
math-verify:    ________
reasoning-gym:  ________
GPU (nvidia-smi): ________  (or "none")

CPU / env:
[ ] compileall grpo
[ ] unittest grpo.math.test_environment
[ ] math reward smoke (correct → 1.0)
[ ] math negative cases (wrong / missing tags → 0.0)
[ ] RG countdown
[ ] RG knights_knaves
[ ] RG shortest_path

Light GPU (optional):
[ ] skipped / PASS / FAIL
[ ] --no-vllm --max-steps 1 countdown smoke
[ ] rollouts.jsonl + adapter saved

Notes / failures:
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named grpo` | `$env:PYTHONPATH = (Get-Location).Path` from repo root |
| `py -3.11` not found | Install Python 3.11, or use `python -m venv .venv` |
| `math-verify` / ANTLR errors | Reinstall `"math-verify[antlr4_13_2]"` |
| RG import / dataset errors | `pip install -r grpo\requirements-reasoning-gym.txt`; check network |
| `torch.cuda.is_available() == False` | Install CUDA PyTorch; update NVIDIA driver; CPU-only → skip section 5 |
| CUDA OOM on GRPO smoke | Skip section 5; use CARC. Env tests (2–4) are still valuable |
| Stale code | `git pull` on `main`; re-run `git rev-parse --short HEAD` |
