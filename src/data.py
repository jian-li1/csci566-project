"""Read test.jsonl only and check its frozen count/hash before inference."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from .io_utils import file_hash, fingerprint
from .shortest_path_grader import grid_problem, shortest_distance, valid_path

TASKS = {"kodcode", "opencode", "competition_math", "shortest_path"}


def _list(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = ast.literal_eval(value)
    if not isinstance(value, list):
        raise ValueError("Expected a JSON/Python list in dataset metadata")
    return value


def load_test(root, run):
    task = run["task"]
    if task not in TASKS:
        raise ValueError(f"Unsupported task {task!r}; choices: {sorted(TASKS)}")
    config = run["data"]
    path = (Path(root) / config["path"]).resolve()
    digest = file_hash(path)
    if digest != config["sha256"]:
        raise ValueError(f"Test SHA-256 mismatch for {path}; use the agreed frozen file")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split", "test") != "test":
            raise ValueError("This evaluator accepts test rows only")
        if task == "kodcode":
            question = row["question"]
            row["test_info"] = _list(row.get("test_info", []))
            row["question_id"] = str(row["question_id"])
            if not isinstance(row["test"], str) or not row["test"].strip():
                raise ValueError("KodCode requires a nonempty test suite")
        elif task == "opencode":
            question = row["input"]
            row["question_id"] = str(row["id"])
            row["unit_tests"] = _list(row["unit_tests"])
            if not row["unit_tests"] or not all(isinstance(t, str) and t.strip() for t in row["unit_tests"]):
                raise ValueError("OpenCode requires a nonempty list of test snippets")
        elif task == "competition_math":
            question = row["problem"]
            # Stable IDs do not depend on row number or the answer.
            row["question_id"] = str(row.get("question_id", "competition_math:" + fingerprint(question)))
            if not isinstance(row["solution"], str) or not row["solution"].strip():
                raise ValueError("Competition Math requires its reference solution")
        else:
            question = row["user"]
            row["question_id"] = str(row.get("question_id", "shortest_path:" + fingerprint(row["matrix"])))
            start, end = grid_problem(row["matrix"])
            distance = shortest_distance(row["matrix"], start, end)
            row.update(start=start, end=end, shortest_distance=distance)
            rendered_grid = "\n".join(" ".join(r) for r in row["matrix"])
            if not question.strip().endswith("\n" + rendered_grid):
                raise ValueError("Shortest Path question and metadata grid differ")
            answer = row["answer"]
            moves = answer.split()
            if distance is None:
                correct_reference = answer == "infeasible"
            else:
                correct_reference = len(moves) == distance and valid_path(row["matrix"], start, end, moves)
            if not correct_reference:
                raise ValueError(f"Incorrect frozen reference for {row['question_id']}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Empty test question")
        row["question_text"] = question
        rows.append(row)
    if len(rows) != config["rows"] or not rows:
        raise ValueError(f"Expected {config['rows']} test rows, found {len(rows)} in {path}")
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate test question IDs")
    if len({" ".join(r["question_text"].split()) for r in rows}) != len(rows):
        raise ValueError("Duplicate test questions")
    return rows
