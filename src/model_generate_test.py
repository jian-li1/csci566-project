from transformermodel import TransformerModel

model = TransformerModel("llama")

messages = [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "4"}
]

print(model.tokenizer.apply_chat_template(
    messages,
    tokenize=False
))

inputs, labels = model.prepare_batch(
    questions=["What is 2 + 2?"],
    answers=["4"]
)

print("Supervised tokens:")
print(model.tokenizer.decode(
    inputs["input_ids"][0][labels[0] != -100],
    skip_special_tokens=False
))
