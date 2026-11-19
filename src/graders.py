"""Task-specific grading behind one interface."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import json
import uuid
from contextlib import suppress
from pathlib import Path

from .math_grader import MathGrader, final_box
from .io_utils import file_hash
from .prompts import extract_code
from .shortest_path_grader import grade as grade_path


def failure(kind, message):
    return {"passed": False, "error_type": kind, "error_message": message, "tests_passed": 0}


class CodeGrader:
    def __init__(self, root, config):
        self.timeout = config.get("timeout_seconds", 20)
        source = str((Path(root) / "src").resolve())
        runtime = config["runtime"]
        self.runtime = runtime
        if not shutil.which(runtime):
            raise RuntimeError(f"{runtime} is unavailable; prepare the code grader as described in README.md")
        if runtime == "docker":
            image = config["image"]
            check = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=20)
            if check.returncode:
                raise RuntimeError(f"Docker image {image!r} is unavailable; run the README build command")
            self.identity = {"runtime": runtime, "image_id": json.loads(check.stdout)[0]["Id"]}
            self.command = [
                "docker", "run", "--rm", "-i", "--network", "none", "--read-only",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m", "--memory", "4g", "--cpus", "1", "--pids-limit", "128",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
                "--env", "OPENBLAS_NUM_THREADS=1", "--env", "OMP_NUM_THREADS=1", "--env", "PYTHONHASHSEED=42",
                "--workdir", "/tmp", "--mount", f"type=bind,src={source},dst=/eval,readonly",
                image, "python", "-I", "-B", "/eval/grade_worker.py",
            ]
        elif runtime == "apptainer":
            image = (Path(root) / config["image"]).resolve()
            if not image.is_file():
                raise FileNotFoundError(f"Apptainer image not found: {image}")
            self.identity = {"runtime": runtime, "image_sha256": file_hash(image)}
            self.command = [
                "apptainer", "exec", "--containall", "--cleanenv", "--no-eval", "--net", "--network", "none",
                "--no-mount", "home,cwd,hostfs,bind-paths", "--pwd", "/tmp",
                "--env", "PYTHONHASHSEED=42,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1",
                "--bind", f"{source}:/eval:ro", str(image), "python", "-I", "-B", "/eval/grade_worker.py",
            ]
            package_dirs = [(Path(root) / p).resolve() for p in config.get("package_dirs", [])]
            for index, path in enumerate(package_dirs):
                if not path.is_dir():
                    raise FileNotFoundError(f"Grader package directory not found: {path}")
                self.command[2:2] = ["--bind", f"{path}:/extra-packages/{index}:ro"]
            self.identity["package_dirs"] = [str(p) for p in package_dirs]
        else:
            raise ValueError("Code grader runtime must be docker or apptainer")

    def score(self, task, code, tests):
        if not code.strip():
            return failure("EmptyCompletion", "No final Python solution was emitted")
        container_name = "project-eval-" + uuid.uuid4().hex if self.runtime == "docker" else None
        command = self.command[:2] + ["--name", container_name] + self.command[2:] if container_name else self.command
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace", start_new_session=os.name != "nt")
        try:
            stdout, stderr = process.communicate(json.dumps({"task": task, "code": code, "tests": tests}), timeout=self.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            if os.name == "nt":
                process.kill()
            else:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            if container_name:
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=15)
            if isinstance(error, KeyboardInterrupt):
                raise
            return failure("Timeout", f"Code tests exceeded {self.timeout} seconds")
        if process.returncode:
            if process.returncode in (125, 126, 127, 255) or "FATAL:" in stderr:
                raise RuntimeError(f"Container grader failed to start: {stderr.strip()[:1000]}")
            return failure("ProcessExit", f"Grader exited with {process.returncode}")
        try:
            result = json.loads(stdout)
            if not isinstance(result.get("passed"), bool):
                raise ValueError("Missing boolean passed field")
            return result
        except (ValueError, AttributeError) as error:
            raise RuntimeError(f"Code grader returned an invalid result: {stdout[:300]} {stderr[:300]}") from error

    def check(self, task):
        tests = "from solution import add\ndef test_add():\n    assert add(1, 2) == 3\n" if task == "kodcode" else ["assert add(1, 2) == 3"]
        good = self.score(task, "def add(a, b): return a + b", tests)
        bad = self.score(task, "def add(a, b): return 0", tests)
        if not good["passed"] or bad["passed"]:
            raise RuntimeError(f"Code grader preflight failed: good={good}, bad={bad}")


class TaskGrader:
    def __init__(self, root, run):
        self.task = run["task"]
        self.code = self.math = None
        self.identity = {"task": self.task, "scorer": "shared-eval-v1"}
        if self.task in {"kodcode", "opencode"}:
            self.code = CodeGrader(root, run["grader"])
            self.code.check(self.task)
            self.identity.update(self.code.identity)
        elif self.task == "competition_math":
            import math_verify  # Check dependencies before starting the worker/GPU.
            import latex2sympy2_extended
            self.math = MathGrader()

    def score(self, row, text):
        if self.code:
            code = extract_code(text)
            tests = row["test"] if self.task == "kodcode" else row["unit_tests"]
            result = self.code.score(self.task, code, tests)
            return {**result, "correct": result["passed"], "score": float(result["passed"]),
                    "extracted_answer": code, "status": "correct" if result["passed"] else result["error_type"]}
        if self.math:
            gold, answer = final_box(row["solution"]), final_box(text)
            if gold is None:
                raise ValueError(f"Missing boxed reference for {row['question_id']}")
            # Exact string equality does not need a subprocess or symbolic parser.
            status = "no_boxed_answer" if answer is None else "exact" if gold == answer else self.math.grade(gold, answer)
            correct = status in {"exact", "equivalent"}
            return {"correct": correct, "score": float(correct), "extracted_answer": answer,
                    "reference_answer": gold, "status": status, "error_type": None if correct else status}
        answer, score, status = grade_path(text, row)
        return {"correct": score == 1.0, "score": score, "extracted_answer": answer,
                "status": status, "error_type": None if score == 1.0 else status}

    def close(self):
        if self.math:
            self.math.close()


def reference_text(task, row):
    if task == "kodcode":
        return row["solution"]
    if task == "opencode":
        return row["output"]
    if task == "competition_math":
        return row["solution"]
    return "<answer>" + row["answer"] + "</answer>"
