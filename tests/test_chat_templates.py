"""Run without model weights: python -m unittest discover -s tests."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chat_templates import configure_chat_template


class ChatTemplateTests(unittest.TestCase):
    def tokenizer(self):
        markers = ("<|begin_of_text|>", "<|start_header_id|>",
                   "<|end_header_id|>", "<|eot_id|>")
        return SimpleNamespace(chat_template=None, get_vocab=lambda: dict.fromkeys(markers))

    def test_llama_prompt_is_prefix_and_answer_has_stop_marker(self):
        tokenizer = self.tokenizer()
        self.assertTrue(configure_chat_template(tokenizer, "llama"))
        template = Environment().from_string(tokenizer.chat_template)
        for system in ([], [{"role": "system", "content": "Solve countdown."}]):
            messages = system + [{"role": "user", "content": "Use 1 and 2 to make 3."}]
            prompt = template.render(messages=messages, bos_token=tokenizer.bos_token,
                                     add_generation_prompt=True)
            full = template.render(messages=messages + [{"role": "assistant", "content": "1+2=3"}],
                                   bos_token=tokenizer.bos_token, add_generation_prompt=False)
            self.assertTrue(prompt.startswith("<|begin_of_text|>"))
            self.assertEqual(full[len(prompt):], "1+2=3<|eot_id|>")
            self.assertTrue(full.startswith(prompt))
            self.assertEqual(tokenizer.eos_token, "<|eot_id|>")

    def test_existing_templates_are_preserved(self):
        tokenizer = SimpleNamespace(chat_template="original")
        self.assertFalse(configure_chat_template(tokenizer, "qwen"))
        self.assertEqual(tokenizer.chat_template, "original")
        self.assertFalse(configure_chat_template(tokenizer, "llama"))

    def test_missing_markers_fail_before_loading_weights(self):
        tokenizer = SimpleNamespace(chat_template=None, get_vocab=lambda: {})
        with self.assertRaisesRegex(ValueError, "markers missing"):
            configure_chat_template(tokenizer, "llama")


if __name__ == "__main__":
    unittest.main()
