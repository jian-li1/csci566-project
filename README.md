# Shared test evaluation

Evaluate an **already trained checkpoint** on our fixed test questions. This package generates one answer per question, applies the task's grader, and saves predictions and metrics. `run_all()` runs the configured evaluations sequentially. There are no training, validation, or checkpoint-selection stages.

## What Advait shares

Share this whole folder in the team repository, including the files under `src/`, `data/`, `configs/`, and `grader/`, plus `run_all.py` and `requirements.txt`. Your teammates supply their own completed checkpoints and edit their paths in a config.

| Location | Contents |
| --- | --- |
| `src/` | Model loading, prompt formatting, test evaluation, answer extraction, graders, and result saving |
| `data/kodcode/test.jsonl` | Our frozen 250-question KodCode test file |
| `data/opencode/test.jsonl` | Our frozen 250-question OpenCode test file |
| `data/competition_math/test.jsonl` | Our frozen 250-question Competition Math test file |
| `data/shortest_path/test.jsonl` | Our frozen 250-question Shortest Path test file |
| `data/manifest.json` | Original test-file counts, hashes, and provenance |
| `configs/` | Example run configurations; copy and edit one per person |
| `checkpoints/` | Each person's completed model or adapter; ignored by Git |
| `grader/` | Container recipes and dependencies for executing code-task tests |
| `outputs/` | Saved results; ignored by Git |
| `run_all.py` | Command-line entry point and import of the public `run_all()` function |

The JSONL files include references/tests needed by the graders. **Only question fields and agreed public prompt metadata go into model inputs.** Reference solutions, gold paths, and unit-test bodies are not included in prompts. KodCode's optional function signatures come from the published `test_info.function_declaration` metadata, as in the Instruct experiment.

The included files are the SFT experiment's fixed test files. Before comparing with GRPO, confirm that those teammates used the same held-out questions and evaluation protocol; this package cannot verify their earlier runs. Do not replace these with a fresh sample or select questions based on a model's test results.

## What each teammate needs to provide

| Required information | Example |
| --- | --- |
| Task | `kodcode`, `opencode`, `competition_math`, or `shortest_path` |
| Checkpoint type and folder | `adapter` plus `/path/to/checkpoint-330`, or `full` plus `/path/to/merged-model` |
| Original base model and exact revision, for adapters | `Qwen/Qwen3.5-4B` and its original commit SHA |
| Model architecture used when saving the adapter | Qwen text model, Qwen full model, or ordinary causal LM |
| Tokenizer and exact input format | Native chat template, explicit Base prompt, or saved raw prompts |
| Evaluation settings | Agreed output-token budget and Qwen thinking mode |

An adapter folder needs **both** `adapter_config.json` and `adapter_model.safetensors` (or `adapter_model.bin`). It also needs access to the original base model. A full/merged checkpoint needs `config.json`, all model-weight shards/index files, and its tokenizer files, or an explicit `tokenizer` path in the config. Trainer-only distributed shards must first be exported into a Transformers/PEFT inference checkpoint by the checkpoint owner.

## 1. Prepare the environment

Run commands from this folder's root. Use an existing **Python 3.11 environment with a working CUDA build of PyTorch** on the GPU machine.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m pip install -r requirements.txt
```

`requirements.txt` pins the reference Transformers/PEFT/math-grader versions used in the supplied experiment. It does not install or replace PyTorch. Use a separate evaluation environment if installing these versions would change an environment you still need. For comparisons, agree on the same installed versions; each run saves its package versions.

For a gated Llama checkpoint, each person needs their own Hugging Face access. Authenticate locally with `hf auth login` if needed. Do not put tokens in configs or Git. For an already cached model, set `offline` to `true`.

## 2. Configure the completed checkpoint

Start with `configs/eval.json`. Edit:

1. `name`: a unique result label, for example `qwen_kodcode_grpo`.
2. `task`: the dataset this checkpoint should answer.
3. `checkpoint.path`: your completed adapter/model folder, relative to this root or absolute.
4. `checkpoint.base_model`, `base_revision`, and `loader`: the exact original architecture and weights for an adapter.
5. `prompt`: the agreed evaluation format for that checkpoint.

The default example is **Qwen Instruct + KodCode + saved LoRA adapter**, using signatures and `enable_thinking: false`. The paths are examples; no model weights are included.

| Example config | Starting point |
| --- | --- |
| `configs/eval.json` | Qwen Instruct adapter, KodCode |
| `configs/eval_carc.example.json` | Same evaluation with an Apptainer grader |
| `configs/qwen_base.example.json` | Qwen Base adapter, explicit KodCode prompt |
| `configs/llama_instruct.example.json` | Llama Instruct adapter |
| `configs/llama_base.example.json` | Llama Base adapter, explicit KodCode prompt |
| `configs/full_model.example.json` | Saved full/merged Llama checkpoint, e.g. GRPO |
| `configs/eval_all.example.json` | Four separate, task-specific completed Qwen checkpoints |

These are starting points, not automatic guesses about your checkpoint. A model trained on one task is not automatically evaluated on every task. For SFT and GRPO on the same task, duplicate a run object inside `runs`, change its `name` and checkpoint fields, and keep the agreed task instructions, thinking mode, signatures, grader, and generation budget identical.

For `checkpoint.loader`, use:

| Value | Architecture |
| --- | --- |
| `auto_causal` | Standard Transformers causal LM, including Llama |
| `qwen3_5_text` | Qwen's `Qwen3_5ForCausalLM` text architecture |
| `qwen3_5_full` | Qwen's full multimodal architecture, used for text inference here |

**An adapter must use the architecture it was saved against.** Our earlier Qwen Base and Competition Math packages use the text loader; the supplied KodCode/OpenCode/Shortest Path Instruct packages use the full loader. The loader checks base-weight compatibility and adapter tensor keys/shapes and stops on a mismatch.

For Base models, `prompt.format` is `base` and `base_template` must match the explicit prompt used for that checkpoint. The included Base examples preserve the KodCode Base prompt. For another task, set its original template explicitly. For Instruct models, use `chat` and the original tokenizer's native template. Keep Qwen's `enable_thinking` explicit. The budget includes any generated reasoning.

If the teammate has exact saved evaluation prompts, `prompt.format: "raw"` reads the `prompt` field of each row. Record the hash/count of that agreed test file in a run-specific `data` object:

```json
"data": {
  "path": "data/my_task/test.jsonl",
  "rows": 250,
  "sha256": "THE_64_CHARACTER_SHA256_OF_THE_AGREED_FILE"
}
```

The row still needs the task's original grading fields. Do not put gold answers into raw prompts. Paths inside all configs resolve relative to the package root. Set an explicit `tokenizer` path when the checkpoint uses a tokenizer saved separately; a remote tokenizer also needs `tokenizer_revision`.

## 3. Prepare the code grader, for KodCode/OpenCode only

Math and Shortest Path do not need a code-execution container.

On a Docker machine, run once from this root:

```bash
docker build -f grader/Dockerfile -t project-eval-grader:1 .
```

The default `grader` setting already uses that image. The recipe installs the scientific/PyTorch CPU packages referenced by the frozen code tasks. Building needs internet access; grading runs with networking disabled.

On CARC, load Apptainer using your allocation's normal module setup. Use `configs/eval_carc.example.json` and set `grader.image` to an available Python 3.11 `.sif`. To build the supplied image where Apptainer builds are supported:

```bash
apptainer build grader/grader.sif grader/apptainer.def
```

An existing compatible `.grader.sif` can also be used. If that image relies on separate Python packages, add their absolute directories under `grader.package_dirs`, for example the old `.grader-packages` folder and the compatible Python 3.11 environment's `site-packages`. They are mounted read-only. Do not mount checkpoint folders. The supplied image already includes its packages.

All compared code runs should use the **same grader image and dependencies**. A newly built image is not automatically identical to a historical grading environment. Check saved package versions and, when comparing with old numbers, replay the reference tests and baseline under the agreed grader. Image identity is recorded. A missing dependency appears as a failure and must be reviewed consistently across methods.

## 4. Run evaluation

Check the config and test-file hashes without loading model weights:

```bash
python run_all.py --config configs/eval.json --check-only
```

This checks data and configuration. It does not test checkpoint loading, CUDA, the tokenizer's native template, or the container runtime.

Run five fixed test questions to check loading, output format, and grading:

```bash
python run_all.py --config configs/eval.json --limit 5
```

Then evaluate **all 250 test questions** for every run in the config:

```bash
python -u run_all.py --config configs/eval.json
```

For CARC, use the edited `configs/eval_carc.example.json` instead. No training or validation command is needed. This evaluates the checkpoint you specify; it does not choose a checkpoint based on test accuracy.

To run only one named entry:

```bash
python -u run_all.py --config configs/eval_all.example.json --run qwen_kodcode_sft
```

To evaluate all task-specific checkpoints after editing their paths:

```bash
python -u run_all.py --config configs/eval_all.example.json
```

If generation and grading use separate machines/environments:

```bash
python -u run_all.py --config configs/eval.json --stage generate
python -u run_all.py --config configs/eval.json --stage grade
```

Keep the same config, source files, weights, and output folder between stages. The grade stage reads saved predictions and does not load GPU weights, though it verifies local checkpoint-file hashes. It requires all requested predictions for a final score. The default `all` stage performs both steps and saves progress after each question.

Optional: check the test references against the current grader, without loading a model:

```bash
python -u run_all.py --config configs/eval.json --stage references
```

Reference-check results live under `outputs/<run>/test/reference-check/`. They report problematic references and do not silently remove questions from the main denominator.

The callable entry point is also available from Python:

```python
from pathlib import Path
from src.evaluate import run_all

results = run_all(Path("configs/eval.json"))
```

## 5. Outputs to send back

For a run named `qwen_kodcode_sft`, send its complete `outputs/qwen_kodcode_sft/test/` folder. It contains:

| File | Purpose |
| --- | --- |
| `summary.json` | Correct count, pass@1, accuracy, failure counts, token-limit hits, runtime |
| `predictions.jsonl` | Question ID, rendered input prompt, raw response, final response, tokens, time |
| `scores.jsonl` | Extracted answer, correctness, task score, and grading status for each prediction |
| `run_config.json` | Model/checkpoint settings, test hash, source hashes, local checkpoint-file hashes |
| `model_load.json` | Actual loader, adapter tensor count, resolved revision, device and GPU |
| `environment.json` | Generation-side package versions |
| `grading_environment.json` | Grading-side package versions and code-grader image identity |

The top-level `outputs/test-summary.csv` summarizes the runs selected in the most recent invocation. Per-run folders remain the authoritative results. Smoke tests write to `outputs/<run>/smoke-5/` and a separate CSV; they never overwrite full test results.

Re-run the same command after an interruption to resume completed questions. Settings, source, test data, and local checkpoint contents are checked before reuse. If any change, use a new run name. Do not run two processes against the same run/output folder at once. Hashing a full local model can take time before inference starts.

The main metric is the fraction of **all requested test questions** answered correctly with one greedy completion. Incomplete runs have no final `pass_at_1`. Code tasks must pass all supplied tests; Math compares the extracted final boxed answers; Shortest Path accepts any shortest valid direction sequence or correctly identified infeasibility. A longer valid path gets task score 0.5 but fails pass@1. Missing final answers, timeouts, and truncated incorrect answers remain in the denominator.

## 6. Add the package to the team repository

Copy these files into your existing cloned team repository, preserving the paths. If it already contains unrelated files, review conflicts before replacing anything. Then, from that repository:

```bash
git add run_all.py README.md requirements.txt .gitignore src configs data grader checkpoints/.gitkeep outputs/.gitkeep
git diff --cached --stat
git commit -m "Add shared test evaluation for completed checkpoints"
git push
```

Commit source, configs, grader recipes, and the fixed test files. Checkpoints, result files, authentication tokens, environments, and `.sif` images stay out of Git. Use `configs/<name>.local.json` for a personal config you do not want committed.

## Package checks and implementation references

The CPU checks exercise frozen data hashes, all task prompt renderers, answer extraction, Shortest Path scoring, checkpoint/config guards, and interrupted-run resume. They use synthetic model responses; they do not establish actual GPU/model compatibility.

```bash
python -m unittest discover -s tests -v
```

The original task graders were carried over from the supplied experiment packages. Model loading follows [Transformers Qwen3.5](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5) and [PEFT inference loading](https://huggingface.co/docs/peft/guides/peft_model_config). GPU inference and the Docker/Apptainer image must be checked on the teammate's machine with the five-question run before the full evaluation.
