"""Chat formatting for Base tokenizers that do not ship a template."""

# Llama 3 role headers and turn boundaries, without synthetic system content.
# All markers already exist in the Llama 3 Base vocabulary.
LLAMA3_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
    " + (message['content'] | trim) + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)


def configure_chat_template(tokenizer, model_name):
    """Keep supplied templates; give Llama Base its family chat format."""
    if tokenizer.chat_template:
        return False
    if model_name != "llama":
        raise ValueError(f"No chat template configured for {model_name}")
    required = ("<|begin_of_text|>", "<|start_header_id|>",
                "<|end_header_id|>", "<|eot_id|>")
    missing = [token for token in required if token not in tokenizer.get_vocab()]
    if missing:
        raise ValueError(f"Llama chat markers missing from tokenizer: {missing}")
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    tokenizer.bos_token = "<|begin_of_text|>"
    tokenizer.eos_token = "<|eot_id|>"
    return True
