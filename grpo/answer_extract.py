"""
Shared final-<answer> extraction for GRPO reward envs.

Used by Reasoning Gym wrappers and Competition Math. Keeps the same format
contract so a mention of <answer> inside <think> cannot pair with the closing tag.
"""

from __future__ import annotations

import re

# Require a final <answer>…</answer> at end-of-string. Body may not contain
# '<' or '>', so a mention of <answer> inside <think> cannot span to the
# closing tag. Do not require a line break before <answer> (same-line
# </think><answer>… is valid).
FINAL_ANSWER_PATTERN = re.compile(
    r"(?is)<answer>[ \t]*"
    r"(?P<body>[^<>]*?)"
    r"[ \t]*</answer>\s*\Z"
)


def extract_answer(response: str) -> tuple[str | None, str | None]:
    """
    Pull the final standalone <answer>...</answer> body from a completion.

    Does NOT check correctness — only format/presence.
    Returns (answer, error). error is None on success.
    """
    if response is None or not str(response).strip():
        return None, "empty response"

    match = FINAL_ANSWER_PATTERN.search(str(response))
    if match is None:
        return None, "missing <answer>...</answer>"
    answer = match.group("body").strip()
    if not answer:
        return None, "empty <answer>"
    return answer, None
