"""
MBPP environment

Goal:
  Write Python code that solves one Mostly Basic Python Problems task.

Action space:

  Agent submits one completion string per episode.

Observation space:
  Agent sees the task id, natural-language problem statement, and a text prompt.
  Reference code, setup code, and test assertions are never shown to the agent.

Internal state:
  Stores the completion, extracted Python code, hidden tests, acceptor flag, and reward.

Step / reward:
  Reward is assigned when the agent submits a completion.
  Verifier is check_code(). Reward is passing tests / total tests,
  unusable action / verifier crash -> None (do not train on this sample).
  Episode terminates after the submit either way.

Failure handling (by severity; set explicitly, not via error-string matching):
  end_episode - invalid/wrong code for this problem (fractional reward in [0, 1])
  end_rollout - empty action / verifier crash (reward None)
  stop_training - fatal infra error (reward None; caller should halt)

Dependencies:
  extract_python_code() - remove an optional Markdown fence from an answer
  check_code()          - run each hidden test in an isolated subprocess

Randomness:
  None. The caller binds a row from huggy-1/mbpp with load_problem().

Security:
  Generated code runs in a resource-limited subprocess with an isolated working
  directory. This contains ordinary crashes and hangs, but is not a security
  boundary. Run training inside a container or VM when model output is untrusted.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

# Error severity -> what the training loop should do.
SEVERITY_END_EPISODE = "end_episode"
SEVERITY_END_ROLLOUT = "end_rollout"
SEVERITY_STOP_TRAINING = "stop_training"

DEFAULT_TEST_TIMEOUT_SECONDS = 2.0
DEFAULT_MEMORY_LIMIT_MB = 512

_WORKER_SOURCE = r"""
import json
import os
import sys

try:
    import resource
except ImportError:
    resource = None

payload = json.loads(sys.stdin.read())
marker = payload.pop("marker")

if resource is not None:
    memory_bytes = int(payload["memory_limit_mb"]) * 1024 * 1024
    cpu_seconds = max(1, int(payload["cpu_limit_seconds"]))
    limits = (
        ("RLIMIT_AS", memory_bytes),
        ("RLIMIT_CPU", cpu_seconds),
        ("RLIMIT_FSIZE", 1024 * 1024),
        ("RLIMIT_NOFILE", 64),
    )
    for limit_name, value in limits:
        limit = getattr(resource, limit_name, None)
        if limit is not None:
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass

namespace = {"__name__": "__mbpp_solution__"}
try:
    setup = payload["setup_code"]
    if setup.strip():
        exec(compile(setup, "<mbpp_setup>", "exec"), namespace, namespace)
    exec(compile(payload["candidate_code"], "<mbpp_candidate>", "exec"), namespace, namespace)
    exec(compile(payload["test_code"], "<mbpp_test>", "exec"), namespace, namespace)
except BaseException as exc:
    error = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:1000]
    sys.stderr.write(error)
    raise SystemExit(1)

os.write(1, ("\n" + marker + "\n").encode("utf-8"))
"""


def extract_python_code(answer: str) -> str:
    """Return Python source from a plain answer or its last Markdown code fence."""
    value = (answer or "").strip()
    fenced = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return fenced[-1].strip() if fenced else value


def _validated_tests(test_list: Sequence[str]) -> list[str]:
    """Normalize hidden tests and reject malformed dataset rows before training."""
    if isinstance(test_list, (str, bytes)) or not isinstance(test_list, Sequence):
        raise TypeError("test_list must be a sequence of assertion strings")

    tests = []
    for index, test in enumerate(test_list):
        if not isinstance(test, str) or not test.strip():
            raise ValueError(f"test_list[{index}] must be a non-empty string")
        try:
            ast.parse(test, mode="exec")
        except SyntaxError as exc:
            raise ValueError(f"test_list[{index}] is invalid Python: {exc}") from exc
        tests.append(test.strip())
    if not tests:
        raise ValueError("test_list must contain at least one test")
    return tests


def _sanitized_environment(temp_dir: str) -> dict[str, str]:
    """Build the minimal environment inherited by a test subprocess."""
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "TMPDIR": temp_dir,
        "TEMP": temp_dir,
        "TMP": temp_dir,
    }
    if "SYSTEMROOT" in os.environ:  # Needed by Python on Windows.
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return environment


def _run_test(
    candidate_code: str,
    test_code: str,
    setup_code: str,
    timeout_seconds: float,
    memory_limit_mb: int,
) -> dict[str, Any]:
    """Run one assertion in a fresh interpreter and return a structured result."""
    marker = f"MBPP_TEST_PASSED_{secrets.token_hex(16)}"
    payload = json.dumps(
        {
            "candidate_code": candidate_code,
            "test_code": test_code,
            "setup_code": setup_code,
            "memory_limit_mb": memory_limit_mb,
            "cpu_limit_seconds": math.ceil(timeout_seconds),
            "marker": marker,
        }
    )

    with tempfile.TemporaryDirectory(prefix="mbpp-test-") as temp_dir:
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _WORKER_SOURCE],
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=temp_dir,
                env=_sanitized_environment(temp_dir),
                start_new_session=(os.name == "posix"),
            )
            try:
                process.communicate(payload.encode("utf-8"), timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
                return {
                    "passed": False,
                    "error": f"timeout after {timeout_seconds:g}s",
                }

            # The candidate may have spawned children. The main worker has already
            # exited, so clean up anything left in its dedicated process group.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            stdout_file.seek(0)
            stdout = stdout_file.read().decode("utf-8", errors="replace")
            passed = process.returncode == 0 and marker in stdout.splitlines()
            if passed:
                return {"passed": True, "error": None}

            stderr_file.seek(0)
            stderr = stderr_file.read(4096).decode("utf-8", errors="replace").strip()
            if process.returncode is not None and process.returncode < 0:
                error = f"terminated by signal {-process.returncode}"
            elif stderr:
                error = stderr.splitlines()[-1]
            else:
                error = f"worker exited with code {process.returncode}"
            return {"passed": False, "error": error}


def check_code(
    candidate_code: str,
    test_list: Sequence[str],
    test_setup_code: str = "",
    timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> dict[str, Any]:
    """
    Check Python code submitted by the agent.

    1) parse the candidate before starting subprocesses
    2) run setup + candidate + one hidden assertion in a fresh subprocess
    3) award one point for each test process that exits successfully
    """
    code = (candidate_code or "").strip()
    tests = _validated_tests(test_list)
    setup = test_setup_code or ""
    if not isinstance(setup, str):
        raise TypeError("test_setup_code must be a string")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if memory_limit_mb <= 0:
        raise ValueError("memory_limit_mb must be positive")

    try:
        ast.parse(setup, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"test_setup_code is invalid Python: {exc}") from exc

    try:
        ast.parse(code, mode="exec")
    except SyntaxError as exc:
        results = [
            {"index": index, "passed": False, "error": f"SyntaxError: {exc.msg}"}
            for index in range(len(tests))
        ]
    else:
        def _score_indexed_test(item: tuple[int, str]) -> dict[str, Any]:
            index, test = item
            result = _run_test(
                code,
                test,
                setup,
                timeout_seconds=timeout_seconds,
                memory_limit_mb=memory_limit_mb,
            )
            return {"index": index, **result}

        # MBPP currently has three tests per row. Running them concurrently keeps
        # one timed-out assertion from imposing three serial timeout windows.
        with ThreadPoolExecutor(max_workers=min(4, len(tests))) as executor:
            results = list(executor.map(_score_indexed_test, enumerate(tests)))

    passed = sum(bool(result["passed"]) for result in results)
    total = len(tests)
    reward = passed / total
    return {
        "correct": passed == total,
        "passed": passed,
        "total": total,
        "reward": reward,
        "tests": results,
        "error": None if passed == total else f"{passed}/{total} tests passed",
    }


class MBPPEnv:
    name = "mbpp"

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
        memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    ) -> None:
        # Goal / observation: implement this natural-language programming task.
        self.task_id: int | None = None
        self.text: str | None = None

        # Hidden verifier data from the dataset row.
        self.test_list: list[str] = []
        self.test_setup_code: str = ""

        # Test-process resource controls.
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb

        # Text prompt for the model (built when a problem is loaded).
        self.prompt: str | None = None

        # Action / internal state.
        self.last_response: str | None = None  # raw completion from the agent
        self.code: str | None = None  # extracted <answer> body (scored source)

        # Episode progress.
        self.current_step: int = 0

        # Internal episode values.
        self.acceptor: bool = False  # whether every hidden test passed
        self.reward: float | None = None  # passing tests / total tests, or None
        self.terminated: bool = False
        self.severity: str | None = None

        self.extract_python_code = extract_python_code
        self.check_code = check_code

    def build_prompt(self) -> str:
        """Turn the current task into a model prompt without leaking tests or code."""
        return (
            "Write Python code that solves the following problem.\n\n"
            f"{self.text}\n\n"
            "Put reasoning in <think>...</think> and the complete executable Python "
            "solution in <answer>...</answer>. The answer may use a Python Markdown "
            "code fence, but must not contain prose outside that fence."
        )

    def state(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "text": self.text,
            "current_step": self.current_step,
            "last_response": self.last_response,
            "terminated": self.terminated,
            "prompt": self.prompt,
        }

    def load_problem(
        self,
        text: str,
        test_list: Sequence[str],
        test_setup_code: str = "",
        task_id: int | None = None,
    ) -> dict[str, Any]:
        """Bind one dataset row and reset all per-episode values."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        tests = _validated_tests(test_list)
        setup = test_setup_code or ""
        if not isinstance(setup, str):
            raise TypeError("test_setup_code must be a string")
        try:
            ast.parse(setup, mode="exec")
        except SyntaxError as exc:
            raise ValueError(f"test_setup_code is invalid Python: {exc}") from exc

        self.task_id = None if task_id is None else int(task_id)
        self.text = text.strip()
        self.test_list = tests
        self.test_setup_code = setup
        self.prompt = self.build_prompt()

        # Reset internal episode values.
        self.last_response = None
        self.code = None
        self.current_step = 0
        self.acceptor = False
        self.reward = None
        self.terminated = False
        self.severity = None

        # Give the agent the start of the problem (metadata + prompt).
        # Note: tests / setup / reference code are intentionally not returned.
        return {
            "task_id": self.task_id,
            "text": self.text,
            "prompt": self.prompt,
        }

    def reset(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Start a new episode from one MBPP dataset row.

        1. Read task_id / text / test_list / test_setup_code from the row.
        2. Validate the hidden assertions and optional setup code.
        3. Reset internal values: acceptor, reward, code.
        4. Return the starting observation (task metadata + prompt).
        """
        if not isinstance(row, dict):
            raise TypeError(f"row must be a dict, got {type(row)}")
        return self.load_problem(
            text=row["text"],
            test_list=row["test_list"],
            test_setup_code=row.get("test_setup_code", ""),
            task_id=row.get("task_id"),
        )

    def step(self, response: str) -> dict[str, Any]:
        """
        Agent submits a completion (reward timing = submit).

        Extracts <answer>...</answer>, then scores with check_code().
        reward: passing tests / total tests; None means unscored (skip train).
        severity: set explicitly for failures (not via error-string matching).

        Rejects step-before-reset and step-after-terminate (reward None).
        """

        def _out(
            result: dict[str, Any], *, reward: float | None | object = ...
        ) -> dict[str, Any]:
            return {
                "observation": {"task_id": self.task_id, "text": self.text},
                "code": self.code,
                "reward": self.reward if reward is ... else reward,
                "acceptor": self.acceptor,
                "terminated": self.terminated,
                "severity": self.severity,
                "check": result,
                "state": self.state(),
            }

        # Episode lifecycle guards - check before mutating scored state.
        if self.text is None or not self.test_list:
            result = {
                "correct": False,
                "passed": 0,
                "total": 0,
                "reward": None,
                "tests": [],
                "error": "step before reset",
            }
            self.reward = None
            self.severity = SEVERITY_END_ROLLOUT
            self.terminated = True
            return _out(result)

        if self.terminated:
            result = {
                "correct": False,
                "passed": 0,
                "total": len(self.test_list),
                "reward": None,
                "tests": [],
                "error": "step after terminate",
            }
            # Keep prior self.reward/code; this return is unscored.
            self.severity = SEVERITY_END_ROLLOUT
            return _out(result, reward=None)

        self.current_step += 1
        self.last_response = response
        self.code = response
        self.acceptor = False
        self.reward = None
        self.severity = None
        self.terminated = True
        result: dict[str, Any]

        if response is None or str(response).strip() == "":
            result = {
                "correct": False,
                "passed": 0,
                "total": len(self.test_list),
                "reward": None,
                "tests": [],
                "error": "empty response",
            }
            self.severity = SEVERITY_END_ROLLOUT
            return _out(result)

        # Prompt asks for <answer>...</answer>; score only that body.
        tags = re.findall(
            r"<answer>(.*?)</answer>", str(response), flags=re.DOTALL | re.IGNORECASE
        )
        if not tags:
            error = "missing <answer>...</answer>"
            answer = ""
        else:
            answer = tags[-1].strip()
            error = "empty <answer>" if not answer else None
        if error is not None:
            result = {
                "correct": False,
                "passed": 0,
                "total": len(self.test_list),
                "reward": 0.0,
                "tests": [],
                "error": error,
            }
            self.reward = 0.0
            self.severity = SEVERITY_END_EPISODE
            return _out(result)

        try:
            self.code = self.extract_python_code(answer)
            result = self.check_code(
                self.code,
                self.test_list,
                self.test_setup_code,
                timeout_seconds=self.timeout_seconds,
                memory_limit_mb=self.memory_limit_mb,
            )
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            result = {
                "correct": False,
                "passed": 0,
                "total": len(self.test_list),
                "reward": None,
                "tests": [],
                "error": str(exc),
            }
            # Typed fatal vs rollout crash - no substring matching on messages.
            if isinstance(exc, (MemoryError, SystemError)):
                self.severity = SEVERITY_STOP_TRAINING
            else:
                self.severity = SEVERITY_END_ROLLOUT
            return _out(result)  # reward stays None

        self.reward = float(result["reward"])
        self.acceptor = bool(result["correct"])
        self.severity = None if self.acceptor else SEVERITY_END_EPISODE
        return _out(result)


if __name__ == "__main__":
    raise SystemExit("mbpp.py is an importable environment; run grpo_mbpp.py to train")
