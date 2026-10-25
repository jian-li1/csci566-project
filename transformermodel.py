from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

class TransformerModel:
    def __init__ (self, model_name:str):
        self.model_name = model_name.lower()
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

    def prepare_batch(self, systems, questions, answers, max_length=4096):
        tokenizer = self.tokenizer

        if systems is None:
            systems = [""] * len(questions)

        full_conversations = []
        prompt_texts = []

        for question, answer, system in zip(
                questions, answers, systems
        ):
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

            full_conversations.append(full_text)
            prompt_texts.append(prompt_text)

        batch_inputs = tokenizer(
            full_conversations,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False
        )

        # Keep prompts unpadded for prefix checking
        prompt_inputs = tokenizer(
            prompt_texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )

        labels = batch_inputs["input_ids"].clone()

        for i, prompt_ids in enumerate(prompt_inputs["input_ids"]):
            prompt_length = len(prompt_ids)
            full_length = int(batch_inputs["attention_mask"][i].sum())

            if prompt_length >= full_length:
                raise ValueError(
                    f"Sample {i}: answer was truncated or is empty"
                )

            # Verify that the separately formatted prompt
            # is actually a prefix of the full conversation
            full_prefix = batch_inputs["input_ids"][i, :prompt_length].tolist()

            if full_prefix != prompt_ids:
                raise ValueError(
                    f"Sample {i}: chat template is not prefix-aligned. "
                    "Use a training-compatible assistant loss mask."
                )

            labels[i, :prompt_length] = -100

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