"""
KodCode environment

Goal:
  Write Python code that solves one KodCode programming task.

Action space:

  Agent submits one completion string per episode.

Observation space:
  Agent sees the question id, natural-language problem statement, and a text prompt.
  Reference code, tests, and reported verification results are never shown to the agent.

Internal state:
  Stores the completion, extracted Python code, hidden tests, acceptor flag, and reward.

Step / reward:
  Reward is assigned when the agent submits a completion.
  Verifier is check_code(). Reward is passing assertions / total assertions,
  unusable action / verifier crash → None (do not train on this sample).
  Episode terminates after the submit either way.

Failure handling (by severity; set explicitly, not via error-string matching):
  end_episode — invalid/wrong code for this problem (fractional reward in [0, 1])
  end_rollout — empty action / verifier crash (reward None)
  stop_training — fatal infra error (reward None; caller should halt)

Dependencies:
  extract_python_code() — remove an optional Markdown fence from an answer
  instrument_test()     — record every assertion inside pytest test functions
  check_code()          — expose generated code as solution.py and run pytest

Randomness:
  None. The caller binds a row from KodCode/KodCode-V1 with load_problem().

Security:
  Generated code runs in a resource-limited subprocess with an isolated working
  directory. This contains ordinary crashes and hangs, but is not a security
  boundary. Run training inside a container or VM when model output is untrusted.
"""

from __future__ import annotations

import ast
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from typing import Any

# Error severity → what the training loop should do.
SEVERITY_END_EPISODE = "end_episode"
SEVERITY_END_ROLLOUT = "end_rollout"
SEVERITY_STOP_TRAINING = "stop_training"

DEFAULT_TEST_TIMEOUT_SECONDS = 10.0
DEFAULT_MEMORY_LIMIT_MB = 2048


class VerifierDependencyError(RuntimeError):
    """Raised when the KodCode verifier itself is missing a required package."""


# Filenames and injected names shared by the parent verifier and pytest worker.
_RECORD_NAME = "__kodcode_record_assertion__"
_BASE_EXCEPTION_NAME = "__kodcode_base_exception__"
_HELPER_MODULE = "_kodcode_assertions"
_RESULT_FILENAME = "assertion_results.json"
_SOLUTION_FILENAME = "solution.py"
_TEST_FILENAME = "test_solution.py"
_RUNNER_FILENAME = "run_kodcode_tests.py"

_HELPER_SOURCE = r"""
import json
import os
import threading

BASE_EXCEPTION = BaseException

_result_path = os.environ["KODCODE_RESULT_PATH"]
_assertion_indices = json.loads(os.environ["KODCODE_ASSERTION_INDICES"])
_outcomes = {str(index): [] for index in _assertion_indices}
_lock = threading.Lock()
_dump = json.dumps
_open = open
_replace = os.replace


def _results(worker_error=None):
    results = []
    for index in _assertion_indices:
        executions = _outcomes[str(index)]
        if not executions:
            results.append(
                {
                    "index": index,
                    "passed": False,
                    "error": worker_error or "assertion was not executed",
                }
            )
            continue

        failures = [execution for execution in executions if not execution["passed"]]
        results.append(
            {
                "index": index,
                "passed": not failures,
                "error": failures[-1]["error"] if failures else None,
            }
        )
    return results


def write_report(worker_error=None, fatal=False):
    payload = _dump(
        {
            "results": _results(worker_error),
            "worker_error": worker_error,
            "fatal": bool(fatal),
        }
    )
    temporary_path = _result_path + ".tmp"
    with _open(temporary_path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    _replace(temporary_path, _result_path)


def record_assertion(index, passed, error):
    with _lock:
        _outcomes[str(index)].append(
            {"passed": bool(passed), "error": None if passed else str(error)[:1000]}
        )
        # Persist after every assertion so partial credit survives a later timeout.
        write_report()


write_report()
"""

_RUNNER_SOURCE = r"""
import math
import os
import sys

try:
    import resource
except ImportError:
    resource = None

root = os.path.dirname(os.path.abspath(__file__))
# ``-I`` omits the script directory from sys.path; add only this temporary sandbox
# so pytest can import solution.py, the test module, and the assertion recorder.
sys.path.insert(0, root)

if resource is not None:
    memory_bytes = int(os.environ["KODCODE_MEMORY_LIMIT_MB"]) * 1024 * 1024
    cpu_seconds = max(1, math.ceil(float(os.environ["KODCODE_CPU_LIMIT_SECONDS"])))
    limits = (
        ("RLIMIT_AS", memory_bytes),
        ("RLIMIT_CPU", cpu_seconds),
        ("RLIMIT_FSIZE", 4 * 1024 * 1024),
        ("RLIMIT_NOFILE", 128),
    )
    for limit_name, value in limits:
        limit = getattr(resource, limit_name, None)
        if limit is not None:
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass

import _kodcode_assertions as recorder

worker_error = None
fatal = False
try:
    import pytest

    exit_code = int(
        pytest.main(
            [
                os.path.join(root, "test_solution.py"),
                "-q",
                "--disable-warnings",
                "-p",
                "no:cacheprovider",
            ]
        )
    )
    if exit_code != 0:
        worker_error = f"pytest exited with code {exit_code}"
except ModuleNotFoundError as exc:
    worker_error = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:1000]
    fatal = exc.name == "pytest"
except BaseException as exc:
    worker_error = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:1000]

recorder.write_report(worker_error, fatal=fatal)
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


def _parse_test(test: str) -> ast.Module:
    """Parse one KodCode pytest module and require at least one assertion."""
    if not isinstance(test, str) or not test.strip():
        raise ValueError("test must be a non-empty string")
    try:
        tree = ast.parse(test, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"test is invalid Python: {exc}") from exc
    if not any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
        raise ValueError("test must contain at least one assertion")
    return tree


class _AssertionInstrumenter(ast.NodeTransformer):
    """Replace assertions with recording try/except blocks."""

    def __init__(self) -> None:
        self.next_index = 0
        self.indices: list[int] = []

    def visit_Assert(self, node: ast.Assert) -> ast.Try:  # noqa: N802
        index = self.next_index
        self.next_index += 1
        self.indices.append(index)
        error_name = f"__kodcode_assertion_error_{index}"

        type_name = ast.Attribute(
            value=ast.Call(
                func=ast.Name(id="type", ctx=ast.Load()),
                args=[ast.Name(id=error_name, ctx=ast.Load())],
                keywords=[],
            ),
            attr="__name__",
            ctx=ast.Load(),
        )
        failure_call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id=_RECORD_NAME, ctx=ast.Load()),
                args=[
                    ast.Constant(index),
                    ast.Constant(False),
                    ast.JoinedStr(
                        values=[
                            ast.FormattedValue(
                                value=type_name,
                                conversion=-1,
                                format_spec=None,
                            ),
                            ast.Constant(": "),
                            ast.FormattedValue(
                                value=ast.Name(id=error_name, ctx=ast.Load()),
                                conversion=-1,
                                format_spec=None,
                            ),
                        ]
                    ),
                ],
                keywords=[],
            )
        )
        success_call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id=_RECORD_NAME, ctx=ast.Load()),
                args=[ast.Constant(index), ast.Constant(True), ast.Constant(None)],
                keywords=[],
            )
        )
        instrumented = ast.Try(
            body=[node],
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name(id=_BASE_EXCEPTION_NAME, ctx=ast.Load()),
                    name=error_name,
                    body=[failure_call],
                )
            ],
            orelse=[success_call],
            finalbody=[],
        )
        return ast.copy_location(instrumented, node)


def _import_insertion_index(body: list[ast.stmt]) -> int:
    """Insert compatibility imports after a docstring and future imports."""
    index = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        index = 1
    while (
        index < len(body)
        and isinstance(body[index], ast.ImportFrom)
        and body[index].module == "__future__"
    ):
        index += 1
    return index


def instrument_test(test: str) -> tuple[str, list[int]]:
    """
    Instrument a KodCode pytest module and make generated symbols importable.

    KodCode usually imports named symbols from ``solution``. Some rows omit that
    import and assume the solution and test share a namespace, so a wildcard import
    is added for compatibility before the recorder bindings are installed.
    """
    tree = _parse_test(test)
    instrumenter = _AssertionInstrumenter()
    tree = instrumenter.visit(tree)

    # Wildcard import supports rows whose tests assume the solution and tests were
    # concatenated. The recorder import follows it so generated names cannot shadow
    # the verifier bindings used by instrumented assertions.
    insert_at = _import_insertion_index(tree.body)
    compatibility_imports: list[ast.stmt] = [
        ast.ImportFrom(module="solution", names=[ast.alias(name="*")], level=0),
        ast.ImportFrom(
            module=_HELPER_MODULE,
            names=[
                ast.alias(name="record_assertion", asname=_RECORD_NAME),
                ast.alias(name="BASE_EXCEPTION", asname=_BASE_EXCEPTION_NAME),
            ],
            level=0,
        ),
    ]
    tree.body[insert_at:insert_at] = compatibility_imports
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), instrumenter.indices


def _sanitized_environment(
    temp_dir: str,
    assertion_indices: list[int],
    timeout_seconds: float,
    memory_limit_mb: int,
) -> dict[str, str]:
    """Build the environment inherited by the pytest subprocess."""
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "TMPDIR": temp_dir,
        "TEMP": temp_dir,
        "TMP": temp_dir,
        "KODCODE_RESULT_PATH": os.path.join(temp_dir, _RESULT_FILENAME),
        "KODCODE_ASSERTION_INDICES": json.dumps(assertion_indices),
        "KODCODE_CPU_LIMIT_SECONDS": str(timeout_seconds),
        "KODCODE_MEMORY_LIMIT_MB": str(memory_limit_mb),
    }
    if "SYSTEMROOT" in os.environ:  # Needed by Python on Windows.
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return environment


def _failed_assertions(
    assertion_indices: list[int], error: str
) -> list[dict[str, Any]]:
    """Build failed results for every assertion in one test module."""
    return [
        {"index": index, "passed": False, "error": error}
        for index in assertion_indices
    ]


def _read_report(
    result_path: str,
    assertion_indices: list[int],
    fallback_error: str,
) -> list[dict[str, Any]]:
    """Read the latest persisted assertion results, or fail every assertion."""
    try:
        with open(result_path, encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return _failed_assertions(assertion_indices, fallback_error)

    results = report.get("results") if isinstance(report, dict) else None
    if isinstance(report, dict) and report.get("fatal"):
        error = report.get("worker_error") or "pytest is required by the verifier"
        raise VerifierDependencyError(str(error))
    if not isinstance(results, list) or len(results) != len(assertion_indices):
        return _failed_assertions(assertion_indices, fallback_error)

    normalized = []
    for index, result in zip(assertion_indices, results):
        if not isinstance(result, dict) or result.get("index") != index:
            return _failed_assertions(assertion_indices, fallback_error)
        passed = bool(result.get("passed", False))
        error = result.get("error")
        if not passed and error == "assertion was not executed":
            error = fallback_error
        normalized.append({"index": index, "passed": passed, "error": error})
    return normalized


def _run_pytest(
    candidate_code: str,
    test_code: str,
    assertion_indices: list[int],
    timeout_seconds: float,
    memory_limit_mb: int,
) -> list[dict[str, Any]]:
    """Write solution.py and run one instrumented KodCode pytest module."""
    with tempfile.TemporaryDirectory(prefix="kodcode-test-") as temp_dir:
        # KodCode tests import from a module literally named ``solution``. Materialize
        # that module beside the instrumented pytest file and private runner helpers.
        paths = {
            _SOLUTION_FILENAME: candidate_code,
            _TEST_FILENAME: test_code,
            f"{_HELPER_MODULE}.py": _HELPER_SOURCE,
            _RUNNER_FILENAME: _RUNNER_SOURCE,
        }
        for filename, content in paths.items():
            with open(os.path.join(temp_dir, filename), "w", encoding="utf-8") as handle:
                handle.write(content)
                if not content.endswith("\n"):
                    handle.write("\n")

        environment = _sanitized_environment(
            temp_dir,
            assertion_indices,
            timeout_seconds,
            memory_limit_mb,
        )
        result_path = environment["KODCODE_RESULT_PATH"]
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                [sys.executable, "-I", os.path.join(temp_dir, _RUNNER_FILENAME)],
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=temp_dir,
                env=environment,
                start_new_session=(os.name == "posix"),
            )
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
                return _read_report(
                    result_path,
                    assertion_indices,
                    f"timeout after {timeout_seconds:g}s",
                )

            # The candidate may have spawned children. The main worker has already
            # exited, so clean up anything left in its dedicated process group.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            stderr_file.seek(0)
            stderr = stderr_file.read(4096).decode("utf-8", errors="replace").strip()
            if process.returncode is not None and process.returncode < 0:
                fallback_error = f"terminated by signal {-process.returncode}"
            elif stderr:
                fallback_error = stderr.splitlines()[-1]
            elif process.returncode:
                fallback_error = f"pytest worker exited with code {process.returncode}"
            else:
                fallback_error = "assertion was not executed"
            return _read_report(result_path, assertion_indices, fallback_error)


def check_code(
    candidate_code: str,
    test: str,
    timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> dict[str, Any]:
    """
    Check Python code submitted by the agent.

    1) parse the candidate and KodCode's pytest-style ``test`` module
    2) expose the generated candidate as an importable ``solution.py`` module
    3) instrument each ``assert`` and run the wrapped test functions with pytest
    4) award passing assertions / total assertions
    """
    code = (candidate_code or "").strip()
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if memory_limit_mb <= 0:
        raise ValueError("memory_limit_mb must be positive")

    instrumented_test, assertion_indices = instrument_test(test)
    try:
        ast.parse(code, mode="exec")
    except SyntaxError as exc:
        results = _failed_assertions(
            assertion_indices,
            f"SyntaxError: {exc.msg}",
        )
    else:
        results = _run_pytest(
            code,
            instrumented_test,
            assertion_indices,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
        )

    passed = sum(bool(result["passed"]) for result in results)
    total = len(results)
    reward = passed / total
    return {
        "correct": passed == total,
        "passed": passed,
        "total": total,
        "reward": reward,
        "tests": results,
        "error": None if passed == total else f"{passed}/{total} assertions passed",
    }


class KodCodeEnv:
    name = "kodcode"

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
        memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    ) -> None:
        # Goal / observation: implement this natural-language programming task.
        self.question_id: str | None = None
        self.question: str | None = None

        # Hidden verifier data from the dataset row.
        self.test: str | None = None
        self.total_assertions: int = 0

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
        self.acceptor: bool = False  # whether every hidden assertion passed
        self.reward: float | None = None  # passing assertions / total, or None
        self.terminated: bool = False
        self.severity: str | None = None

        self.extract_python_code = extract_python_code
        self.instrument_test = instrument_test
        self.check_code = check_code

    def build_prompt(self) -> str:
        """Turn the current question into a model prompt without leaking tests."""
        return (
            "Write Python code that solves the following problem.\n\n"
            f"{self.question}\n\n"
            "Put step-by-step reasoning in <think>...</think> and the complete "
            "executable Python solution in <answer>...</answer>. The answer may use "
            "a Python Markdown code fence, but must not contain prose outside it."
        )

    def state(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "current_step": self.current_step,
            "last_response": self.last_response,
            "terminated": self.terminated,
            "prompt": self.prompt,
        }

    def load_problem(
        self,
        problem: dict[str, Any] | str,
        test: str | None = None,
        question_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Bind one dataset row and reset all per-episode values.

        The preferred form matches ``make_env_reward_fn``::

            env.load_problem({
                "question_id": "...",
                "question": "...",
                "test": "...",
            })

        A positional form is also available for standalone callers::

            env.load_problem(question, test, question_id)
        """
        if isinstance(problem, dict):
            if test is not None or question_id is not None:
                raise TypeError(
                    "when problem is a dict, do not pass test or question_id"
                )
            row = problem
            problem = row["question"]
            test = row["test"]
            question_id = row.get("question_id")

        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("question must be a non-empty string")
        if test is None:
            raise ValueError("test is required")
        test_tree = _parse_test(test)

        self.question_id = None if question_id is None else str(question_id)
        self.question = problem.strip()
        self.test = test.strip()
        self.total_assertions = sum(
            isinstance(node, ast.Assert) for node in ast.walk(test_tree)
        )
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
        # Note: tests / reference code / reported results are intentionally omitted.
        return {
            "question_id": self.question_id,
            "question": self.question,
            "prompt": self.prompt,
        }

    def reset(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Start a new episode from one KodCode dataset row.

        1. Read question_id / question / test from the row.
        2. Parse and validate the hidden pytest module.
        3. Reset internal values: acceptor, reward, code.
        4. Return the starting observation (task metadata + prompt).
        """
        if not isinstance(row, dict):
            raise TypeError(f"row must be a dict, got {type(row)}")
        return self.load_problem(row)

    def step(self, response: str) -> dict[str, Any]:
        """
        Agent submits a completion (reward timing = submit).

        Extracts <answer>...</answer>, then scores with check_code().
        reward: passing assertions / total assertions; None means unscored.
        severity: set explicitly for failures (not via error-string matching).

        Rejects step-before-reset and step-after-terminate (reward None).
        """

        def _out(
            result: dict[str, Any], *, reward: float | None | object = ...
        ) -> dict[str, Any]:
            return {
                "observation": {
                    "question_id": self.question_id,
                    "question": self.question,
                },
                "code": self.code,
                "reward": self.reward if reward is ... else reward,
                "acceptor": self.acceptor,
                "terminated": self.terminated,
                "severity": self.severity,
                "check": result,
                "state": self.state(),
            }

        # Episode lifecycle guards — check before mutating scored state.
        if self.question is None or self.test is None:
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
                "total": self.total_assertions,
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
                "total": self.total_assertions,
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
                "total": self.total_assertions,
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
                self.test,
                timeout_seconds=self.timeout_seconds,
                memory_limit_mb=self.memory_limit_mb,
            )
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            result = {
                "correct": False,
                "passed": 0,
                "total": self.total_assertions,
                "reward": None,
                "tests": [],
                "error": str(exc),
            }
            # Typed fatal vs rollout crash — no substring matching on messages.
            if isinstance(exc, (MemoryError, SystemError, VerifierDependencyError)):
                self.severity = SEVERITY_STOP_TRAINING
            else:
                self.severity = SEVERITY_END_ROLLOUT
            return _out(result)  # reward stays None

        self.reward = float(result["reward"])
        self.acceptor = bool(result["correct"])
        self.severity = None if self.acceptor else SEVERITY_END_EPISODE
        return _out(result)
