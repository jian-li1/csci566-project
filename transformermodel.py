from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Subset

class TransformerModel:
    def __init__ (self, model_name:str, max_length:int = 4096):
        self.model_name = model_name.lower()
        self.max_length = max_length
        if self.model_name == "llama":
            model_id = "meta-llama/Llama-3.2-3B-Instruct"
            model_loader = AutoModelForCausalLM

        elif self.model_name == "qwen":
            model_id = "Qwen/Qwen3.5-4B"
            model_loader = AutoModelForMultimodalLM

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        # Load Model
        self.model = model_loader.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(
            self.model,
            lora_config
        )

        self.model.print_trainable_parameters()

        trainable_params = [
            p for p in self.model.parameters()
            if p.requires_grad
        ]

    def tokenize_example(self, question, answer, system=""):
        """Render one sample with the chat template and tokenize it, untruncated.

        Returns (full_ids, prompt_length). Raises only on a chat-template
        problem, which is a code bug rather than a data problem.
        """
        tokenizer = self.tokenizer

        prompt_messages = []

        # Only add system when it exists
        if system and system.strip():
            prompt_messages.append({
                "role": "system",
                "content": system
            })

        prompt_messages.append({
            "role": "user",
            "content": question
        })

        full_messages = prompt_messages + [
            {"role": "assistant", "content": answer}
        ]

        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Verify that the separately formatted prompt
        # is actually a prefix of the full conversation (as text)
        if not full_text.startswith(prompt_text):
            raise ValueError(
                "Prompt text is not a prefix of the full conversation. "
                "Check the model's chat template."
            )

        # Tokenize prompt and completion separately and join them, so the
        # prompt tokens are exactly what the model sees at inference. Tokenizing
        # full_text in one go can merge tokens across the boundary (Qwen's
        # "<think>\n" + "\n</think>" becomes a single "\n\n" token).
        # No truncation: we need the real length to decide whether to skip
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False
        )["input_ids"]

        completion_ids = tokenizer(
            full_text[len(prompt_text):],
            add_special_tokens=False,
            truncation=False
        )["input_ids"]

        return prompt_ids + completion_ids, len(prompt_ids)

    def filter_dataset(self, dataset, name="dataset"):
        """One pass before training: skip samples that are too long or have
        an empty answer, and report how many were skipped."""
        keep = []
        too_long = 0
        empty = 0
        lengths = []

        for idx in range(len(dataset)):
            ex = dataset[idx]
            answer = ex["answer"]

            # Checked on the text: an empty answer still renders an
            # end-of-turn token, so a token-count check would miss it
            if answer is None or not str(answer).strip():
                empty += 1
                continue

            full_ids, _ = self.tokenize_example(
                ex["question"], answer, ex.get("system", "")
            )
            lengths.append(len(full_ids))

            if len(full_ids) > self.max_length:
                too_long += 1
            else:
                keep.append(idx)

        print(
            f"[{name}] kept {len(keep)}/{len(dataset)} | "
            f"skipped too long (>{self.max_length} tokens): {too_long} | "
            f"skipped empty answer: {empty}"
        )
        if lengths:
            s = sorted(lengths)
            print(
                f"[{name}] token lengths: p50={s[len(s) // 2]}, "
                f"p95={s[int(len(s) * 0.95)]}, max={s[-1]}"
            )
        if not keep:
            raise ValueError(f"[{name}] no usable samples left after filtering")

        return Subset(dataset, keep)

    def prepare_batch(self, systems, questions, answers):
        tokenizer = self.tokenizer

        if systems is None:
            systems = [""] * len(questions)

        if not (len(systems) == len(questions) == len(answers)):
            raise ValueError("Batch fields have different lengths")

        full_token_lists = []
        prompt_lengths = []

        for i, (question, answer, system) in enumerate(
                zip(questions, answers, systems)
        ):
            full_ids, prompt_length = self.tokenize_example(
                question, answer, system
            )

            # Safety net only: filter_dataset already removed these
            if len(full_ids) > self.max_length or len(full_ids) <= prompt_length:
                raise RuntimeError(
                    f"Sample {i} slipped past filter_dataset "
                    f"(len={len(full_ids)}, prompt={prompt_length}, "
                    f"max={self.max_length})"
                )

            full_token_lists.append(full_ids)
            prompt_lengths.append(prompt_length)

        # Pad to the longest sequence in the batch (right padding)
        batch_inputs = tokenizer.pad(
            [
                {"input_ids": ids, "attention_mask": [1] * len(ids)}
                for ids in full_token_lists
            ],
            padding=True,
            return_tensors="pt"
        )

        labels = batch_inputs["input_ids"].clone()

        # Ignore system + user + assistant header
        for i, prompt_length in enumerate(prompt_lengths):
            labels[i, :prompt_length] = -100

        # Ignore padding (masked by attention_mask, so real EOS tokens stay trained)
        labels[batch_inputs["attention_mask"] == 0] = -100

        batch_inputs = {
            key: value.to(self.model.device)
            for key, value in batch_inputs.items()
        }

        labels = labels.to(self.model.device)

        return batch_inputs, labels

    def calculate_batch_loss(self, questions, answers, systems = None):
        inputs, labels = self.prepare_batch(questions= questions ,answers=answers,systems=systems)
        output = self.model(**inputs, labels = labels)
        return output.loss

    def generate(self,
            question: str,
            max_new_tokens=4096,
            system: str = ""
    ):
        self.model.eval()

        messages = []

        if system and system.strip():
            messages.append({
                "role": "system",
                "content": system
            })

        messages.append({
            "role": "user",
            "content": question
        })

        # Keep existing generation code

    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)