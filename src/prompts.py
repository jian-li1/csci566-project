"""Only public task fields reach the model; gold answers/tests are for grading."""
from __future__ import annotations

import ast
import re

KODCODE_SYSTEM = (
    "Solve the Python task. Preserve the requested function and class names and their parameters. "
    "Include all imports and helper definitions needed to run the solution. "
    "Return the complete solution in one fenced Python code block."
)
MATH_PREFIX = (
    "Solve the following competition math problem. Show your work and put your final answer "
    r"in \boxed{...}." + "\n\n"
)


def missing_signatures(row):
    signatures = []
    for metadata in row["test_info"]:
        name, declaration = metadata["function_name"], metadata["function_declaration"].strip()
        if re.search(rf"\bdef\s+{re.escape(name)}\s*\(", row["question"]):
            continue
        functions = ast.parse(declaration + "\n    pass\n").body
        if len(functions) != 1 or not isinstance(functions[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"Invalid published function signature for {row['question_id']}")
        function = functions[0]
        if function.name != name or function.decorator_list or "\n" in declaration or len(function.body) != 1 or not isinstance(function.body[0], ast.Pass):
            raise ValueError(f"Invalid published function signature for {row['question_id']}")
        signatures.append(declaration)
    return list(dict.fromkeys(signatures))


def task_text(row, run):
    task, settings = run["task"], run["prompt"]
    system, question = "", row["question_text"]
    if task == "kodcode":
        if settings["format"] == "chat":
            system, question = KODCODE_SYSTEM, question.strip()
        else:
            question = question.rstrip()
        if settings.get("include_signatures", False):
            signatures = missing_signatures(row)
            if signatures:
                question += "\n\nAdditional function signatures supplied by KodCode:\n```python\n" + "\n".join(signatures) + "\n```"
    elif task == "competition_math":
        question = MATH_PREFIX + question
    elif task == "shortest_path":
        system, question = row["system"].strip(), row["user"].strip()
    if settings.get("system_prompt") is not None:
        system = settings["system_prompt"]
    return system, question


def render_prompt(tokenizer, row, run):
    settings = run["prompt"]
    if settings["format"] == "raw":
        # Teammates can supply their exact saved evaluation prompt, including special tokens.
        text = row.get("prompt")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("raw prompt format requires a nonempty prompt field in every test row")
    else:
        system, question = task_text(row, run)
        if settings["format"] == "chat":
            if not tokenizer.chat_template:
                raise ValueError("The tokenizer has no chat template; choose the format used for this checkpoint")
            messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 **settings.get("chat_template_kwargs", {}))
        else:
            text = settings["base_template"].format(system=system, question=question)
    text += settings.get("assistant_prefill", "")
    ids = tokenizer.encode(text, add_special_tokens=False)
    if settings.get("add_bos", False):
        if tokenizer.bos_token_id is None:
            raise ValueError("add_bos is true but this tokenizer has no BOS token")
        ids = [tokenizer.bos_token_id, *ids]
    return text, ids


def final_text(raw, prompt_text):
    """Require the reasoning section to finish when the prompt opened <think>."""
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[-1].strip(), False
    open_in_prompt = prompt_text.rfind("<think>") > prompt_text.rfind("</think>")
    if open_in_prompt or "<think>" in raw:
        return "", True
    return raw.strip(), False


def extract_code(text):
    fences = re.findall(r"^[ \t]*```([^\r\n`]*)\r?\n(.*?)(?:^[ \t]*```[ \t]*\r?$|\Z)", text, re.M | re.S)
    for language, code in fences:
        if language.strip().lower() in {"", "py", "python", "python3"} and code.strip():
            return code.strip("\r\n")
    return text.strip("\r\n")
