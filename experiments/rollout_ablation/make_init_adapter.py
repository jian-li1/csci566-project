"""
Save the one LoRA initialization every ablation arm starts from.

    python -m experiments.rollout_ablation.make_init_adapter --out runs/ablation/init_adapter/seed0

Uses grpo.grpo.MODEL_ID and grpo.grpo.build_lora_config(), so the adapter is the
same one grpo.grpo would have created itself, just with a pinned seed and a hash.
Writes <out>/init_adapter.json with model id, seed, LoRA config and sha256.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_init_adapter(out: Path, seed: int, model_id: str | None = None) -> dict:
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    import grpo.grpo as g

    model_id = model_id or g.MODEL_ID
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    torch.manual_seed(seed)
    model = get_peft_model(base, g.build_lora_config())
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    info = {
        "model_id": model_id,
        "seed": seed,
        "lora_config": {k: (sorted(v) if isinstance(v, set) else v)
                        for k, v in g.build_lora_config().to_dict().items()},
        "adapter_sha256": sha256_file(out / "adapter_model.safetensors"),
    }
    (out / "init_adapter.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-id", default=None, help="default grpo.grpo.MODEL_ID")
    p.add_argument("--if-missing", action="store_true", help="no-op if init_adapter.json exists")
    a = p.parse_args(argv)
    if a.if_missing and (a.out / "init_adapter.json").is_file():
        print(f"init adapter exists: {a.out}")
        return 0
    info = make_init_adapter(a.out, a.seed, a.model_id)
    print(f"wrote {a.out} sha256={info['adapter_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
