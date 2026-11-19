"""Load an existing full checkpoint or PEFT adapter for inference only."""
from __future__ import annotations

import gc
import time
import warnings
from pathlib import Path

from .prompts import final_text, render_prompt


def local_or_hub(root, value):
    candidate = Path(root) / value
    return str(candidate.resolve()) if candidate.exists() else value


class ModelRunner:
    def __init__(self, root, run, generation):
        import torch
        import transformers

        self.torch, self.run, self.generation = torch, run, generation
        checkpoint = run["checkpoint"]
        adapter = checkpoint["type"] == "adapter"
        source = checkpoint["base_model"] if adapter else checkpoint["path"]
        source = local_or_hub(root, source)
        revision = checkpoint.get("base_revision") if adapter else checkpoint.get("revision")
        common = {"trust_remote_code": False, "local_files_only": run.get("offline", False)}
        if revision:
            common["revision"] = revision
        tokenizer_source = run.get("tokenizer", source)
        tokenizer_common = dict(common)
        if run.get("tokenizer"):
            tokenizer_common.pop("revision", None)
            if run.get("tokenizer_revision"):
                tokenizer_common["revision"] = run["tokenizer_revision"]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(local_or_hub(root, tokenizer_source), **tokenizer_common)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer is missing EOS")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device(run.get("device", "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; activate the GPU environment or set device to cpu for a small check")
        dtype = getattr(torch, run.get("dtype", "bfloat16"))
        if self.device.type == "cpu" and dtype != torch.float32:
            raise ValueError("Use dtype=float32 for CPU evaluation")
        if self.device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bfloat16; use the agreed float16 setting")
        config = transformers.AutoConfig.from_pretrained(source, **common)
        loader_name = checkpoint["loader"]
        if loader_name == "qwen3_5_text":
            loader = getattr(transformers, "Qwen3_5ForCausalLM")
            config = config.get_text_config()
        elif loader_name == "qwen3_5_full":
            loader = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForImageTextToText
        else:
            loader = transformers.AutoModelForCausalLM
        self.model, info = loader.from_pretrained(
            source, config=config, dtype=dtype, device_map={"": self.device},
            attn_implementation="sdpa", use_safetensors=True, output_loading_info=True, **common,
        )
        problems = {key: info[key] for key in ("missing_keys", "mismatched_keys", "error_msgs") if info.get(key)}
        unexpected = info.get("unexpected_keys", [])
        if loader_name == "qwen3_5_text":
            unexpected = [key for key in unexpected if not key.startswith(("model.visual.", "visual.", "mtp.", "model.mtp."))]
        if unexpected:
            problems["unexpected_keys"] = unexpected
        if problems:
            raise RuntimeError(f"Checkpoint architecture/weights do not match the configured loader: {problems}")
        adapter_info = None
        if adapter:
            from peft import PeftConfig, PeftModel, get_peft_model_state_dict
            from peft.utils.save_and_load import load_peft_weights

            path = str((Path(root) / checkpoint["path"]).resolve())
            peft_config = PeftConfig.from_pretrained(path)
            expected_base = checkpoint["base_model"].rstrip("/")
            if peft_config.base_model_name_or_path.rstrip("/") != expected_base:
                raise ValueError(f"Adapter base is {peft_config.base_model_name_or_path!r}; config specifies {expected_base!r}")
            if peft_config.revision and revision and peft_config.revision != revision:
                raise ValueError("Adapter and configured base revisions differ")
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                self.model = PeftModel.from_pretrained(self.model, path, is_trainable=False)
            for warning in recorded:
                message = str(warning.message)
                if "missing adapter" in message.lower() or "mismatch" in message.lower():
                    raise RuntimeError(f"Adapter load failed: {message}")
                warnings.warn(message, stacklevel=2)
            saved, actual = load_peft_weights(path, device="cpu"), get_peft_model_state_dict(self.model)
            missing, extra = sorted(set(actual) - set(saved)), sorted(set(saved) - set(actual))
            shapes = [key for key in set(saved) & set(actual) if saved[key].shape != actual[key].shape]
            if missing or extra or shapes:
                raise RuntimeError(f"Adapter keys do not match this architecture: missing={missing[:8]}, extra={extra[:8]}, shape_mismatches={shapes[:8]}")
            adapter_info = {"loaded_tensor_count": len(saved), "base_revision_in_adapter": peft_config.revision}
            del saved, actual
        self.model.requires_grad_(False)
        self.model.eval()
        text_config = self.model.config.get_text_config()
        self.context_limit = getattr(text_config, "max_position_embeddings", None)
        stops = set()
        for value in (self.model.generation_config.eos_token_id, self.tokenizer.eos_token_id):
            stops.update(value if isinstance(value, list) else [] if value is None else [value])
        for token in ("<|im_end|>", "<|eot_id|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id in self.tokenizer.all_special_ids and token_id != self.tokenizer.unk_token_id:
                stops.add(token_id)
        self.stop_ids = sorted(stops)
        self.generation_config = transformers.GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=generation["max_new_tokens"],
            use_cache=True, eos_token_id=self.stop_ids, pad_token_id=self.tokenizer.pad_token_id,
        )
        self.metadata = {
            "resolved_base_revision": getattr(config, "_commit_hash", None) or revision,
            "loader": type(self.model).__name__, "tokenizer": tokenizer_source,
            "eos_token_ids": self.stop_ids, "context_limit": self.context_limit,
            "adapter": adapter_info,
            "device": str(self.device), "dtype": str(dtype),
            "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
        }
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def generate(self, row):
        torch = self.torch
        prompt_text, prompt_ids = render_prompt(self.tokenizer, row, self.run)
        if not prompt_ids:
            raise ValueError("Empty tokenized prompt")
        if self.context_limit and len(prompt_ids) + self.generation["max_new_tokens"] > self.context_limit:
            raise ValueError(f"{row['question_id']}: prompt plus output budget exceeds the model context; prompts are never silently truncated")
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=self.generation_config)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        generated = output[0, len(prompt_ids):].tolist()
        # Preserve reasoning tags, while removing EOS/EOT from the text sent to graders.
        content_ids = generated[:-1] if generated and generated[-1] in self.stop_ids else generated
        raw = self.tokenizer.decode(content_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        text, unfinished = final_text(raw, prompt_text)
        return {
            "question_id": row["question_id"], "prompt": prompt_text, "raw_response": raw,
            "final_text": text, "unfinished_reasoning": unfinished,
            "prompt_tokens": len(prompt_ids), "generated_tokens": len(generated),
            "hit_token_limit": len(generated) >= self.generation["max_new_tokens"] and (not generated or generated[-1] not in self.stop_ids),
            "generation_seconds": seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30 if self.device.type == "cuda" else None,
        }

    def close(self):
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
