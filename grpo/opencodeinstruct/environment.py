"""
OpenCodeInstruct environment

Goal:
  Write Python code that solves one OpenCodeInstruct programming task.

Action space:

  Agent submits one completion string per episode.

Observation space:
  Agent sees the example id, natural-language problem statement, and a text prompt.
  Reference code, unit tests, and reported test results are never shown to the agent.

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
  parse_unit_tests()    — safely parse the dataset's serialized test list
  extract_python_code() — remove an optional Markdown fence from an answer
  check_code()          — run hidden assertions in isolated subprocesses

Randomness:
  None. The caller binds a row from NVIDIA/OpenCodeInstruct with load_problem().

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

# Error severity → what the training loop should do.
SEVERITY_END_EPISODE = "end_episode"
SEVERITY_END_ROLLOUT = "end_rollout"
SEVERITY_STOP_TRAINING = "stop_training"

DEFAULT_TEST_TIMEOUT_SECONDS = 2.0
DEFAULT_MEMORY_LIMIT_MB = 512
DEFAULT_MAX_TEST_WORKERS = 4

_RECORD_NAME = "__opencodeinstruct_record_assertion__"
_BASE_EXCEPTION_NAME = "__opencodeinstruct_base_exception__"

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
write_result = os.write
dump_result = json.dumps

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

assertion_indices = payload["assertion_indices"]
outcomes = {str(index): [] for index in assertion_indices}


def record_assertion(index, passed, error):
    outcomes[str(index)].append(
        {"passed": bool(passed), "error": None if passed else str(error)[:1000]}
    )


namespace = {"__name__": "__opencodeinstruct_solution__"}
worker_error = None
try:
    exec(
        compile(payload["candidate_code"], "<opencodeinstruct_candidate>", "exec"),
        namespace,
        namespace,
    )
    namespace[payload["record_name"]] = record_assertion
    namespace[payload["base_exception_name"]] = BaseException
    exec(
        compile(payload["test_code"], "<opencodeinstruct_test>", "exec"),
        namespace,
        namespace,
    )
except BaseException as exc:
    worker_error = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:1000]

results = []
for index in assertion_indices:
    executions = outcomes[str(index)]
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

report = dump_result({"results": results, "worker_error": worker_error})
write_result(1, ("\n" + marker + report + "\n").encode("utf-8"))
"""


def parse_unit_tests(value: str | Sequence[str]) -> list[str]:
    """Parse and validate a list of OpenCodeInstruct unit-test snippets."""
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError, TypeError) as exc:
            raise ValueError("unit_tests is not a valid serialized list") from exc
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parsed = list(value)
    else:
        raise TypeError("unit_tests must be a serialized list or sequence of strings")

    if not isinstance(parsed, list) or not parsed:
        raise ValueError("unit_tests must contain at least one test snippet")

    tests = []
    assertion_count = 0
    for index, test in enumerate(parsed):
        if not isinstance(test, str) or not test.strip():
            raise ValueError(f"unit_tests[{index}] must be a non-empty string")
        try:
            tree = ast.parse(test, mode="exec")
        except SyntaxError as exc:
            raise ValueError(f"unit_tests[{index}] is invalid Python: {exc}") from exc
        assertion_count += sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
        tests.append(test.strip())

    if assertion_count == 0:
        raise ValueError("unit_tests must contain at least one assertion")
    return tests


def extract_python_code(answer: str) -> str:
    """Return Python source from a plain answer or its last Markdown code fence."""
    value = (answer or "").strip()
    fenced = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return fenced[-1].strip() if fenced else value


def _count_assertions(tests: Sequence[str]) -> int:
    """Count static ``assert`` statements across validated test snippets."""
    return sum(
        isinstance(node, ast.Assert)
        for test in tests
        for node in ast.walk(ast.parse(test, mode="exec"))
    )


class _AssertionInstrumenter(ast.NodeTransformer):
    """Replace assertions with recording try/except blocks."""

    def __init__(self, start_index: int) -> None:
        self.next_index = start_index
        self.indices: list[int] = []

    def visit_Assert(self, node: ast.Assert) -> ast.Try:  # noqa: N802
        index = self.next_index
        self.next_index += 1
        self.indices.append(index)
        error_name = f"__opencodeinstruct_assertion_error_{index}"

        failure_call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id=_RECORD_NAME, ctx=ast.Load()),
                args=[
                    ast.Constant(index),
                    ast.Constant(False),
                    ast.JoinedStr(
                        values=[
                            ast.FormattedValue(
                                value=ast.Call(
                                    func=ast.Name(id="type", ctx=ast.Load()),
                                    args=[ast.Name(id=error_name, ctx=ast.Load())],
                                    keywords=[],
                                ),
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
        # ``type(exc)`` above would format as ``<class 'AssertionError'>``.
        # Replace it with ``type(exc).__name__`` for concise diagnostics.
        type_call = failure_call.value.args[2].values[0].value
        failure_call.value.args[2].values[0].value = ast.Attribute(
            value=type_call,
            attr="__name__",
            ctx=ast.Load(),
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


def _instrument_test(test: str, start_index: int) -> tuple[str, list[int]]:
    """Instrument every static assertion in one unit-test snippet."""
    tree = ast.parse(test, mode="exec")
    instrumenter = _AssertionInstrumenter(start_index)
    tree = instrumenter.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), instrumenter.indices


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


def _failed_assertions(
    assertion_indices: Sequence[int], error: str
) -> list[dict[str, Any]]:
    """Build failed results for every assertion in one test snippet."""
    return [
        {"index": index, "passed": False, "error": error}
        for index in assertion_indices
    ]


def _run_test(
    candidate_code: str,
    test_code: str,
    assertion_indices: Sequence[int],
    timeout_seconds: float,
    memory_limit_mb: int,
) -> list[dict[str, Any]]:
    """Run one instrumented test snippet in a fresh interpreter."""
    marker = f"OPENCODEINSTRUCT_RESULTS_{secrets.token_hex(16)}:"
    payload = json.dumps(
        {
            "candidate_code": candidate_code,
            "test_code": test_code,
            "assertion_indices": list(assertion_indices),
            "memory_limit_mb": memory_limit_mb,
            "cpu_limit_seconds": math.ceil(timeout_seconds),
            "record_name": _RECORD_NAME,
            "base_exception_name": _BASE_EXCEPTION_NAME,
            "marker": marker,
        }
    )

    with tempfile.TemporaryDirectory(prefix="opencodeinstruct-test-") as temp_dir:
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", _WORKER_SOURCE],
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
                return _failed_assertions(
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

            stdout_file.seek(0)
            stdout = stdout_file.read().decode("utf-8", errors="replace")
            report_line = next(
                (line for line in reversed(stdout.splitlines()) if line.startswith(marker)),
                None,
            )
            if report_line is not None:
                try:
                    report = json.loads(report_line[len(marker) :])
                except json.JSONDecodeError:
                    report = None
                if isinstance(report, dict) and isinstance(report.get("results"), list):
                    return report["results"]

            stderr_file.seek(0)
            stderr = stderr_file.read(4096).decode("utf-8", errors="replace").strip()
            if process.returncode is not None and process.returncode < 0:
                error = f"terminated by signal {-process.returncode}"
            elif stderr:
                error = stderr.splitlines()[-1]
            else:
                error = f"worker exited with code {process.returncode}"
            return _failed_assertions(assertion_indices, error)


def check_code(
    candidate_code: str,
    unit_tests: str | Sequence[str],
    timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    max_test_workers: int = DEFAULT_MAX_TEST_WORKERS,
) -> dict[str, Any]:
    """
    Check Python code submitted by the agent.

    1) parse the candidate and the dataset's serialized unit-test list
    2) instrument each ``assert`` so every assertion receives its own result
    3) run each assertion-bearing snippet in an isolated subprocess
    4) award passing assertions / total assertions
    """
    code = (candidate_code or "").strip()
    tests = parse_unit_tests(unit_tests)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if memory_limit_mb <= 0:
        raise ValueError("memory_limit_mb must be positive")
    if max_test_workers <= 0:
        raise ValueError("max_test_workers must be positive")

    instrumented_tests: list[tuple[int, str, list[int]]] = []
    next_index = 0
    for unit_test_index, test in enumerate(tests):
        instrumented, indices = _instrument_test(test, next_index)
        next_index += len(indices)
        if indices:
            instrumented_tests.append((unit_test_index, instrumented, indices))

    try:
        ast.parse(code, mode="exec")
    except SyntaxError as exc:
        results = _failed_assertions(
            range(next_index),
            f"SyntaxError: {exc.msg}",
        )
        unit_test_by_assertion = {
            index: unit_test_index
            for unit_test_index, _, indices in instrumented_tests
            for index in indices
        }
        results = [
            {**result, "unit_test_index": unit_test_by_assertion[result["index"]]}
            for result in results
        ]
    else:
        def _score_test(item: tuple[int, str, list[int]]) -> list[dict[str, Any]]:
            unit_test_index, test, indices = item
            results = _run_test(
                code,
                test,
                indices,
                timeout_seconds=timeout_seconds,
                memory_limit_mb=memory_limit_mb,
            )
            return [
                {**result, "unit_test_index": unit_test_index}
                for result in results
            ]

        workers = min(max_test_workers, len(instrumented_tests))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            grouped_results = list(executor.map(_score_test, instrumented_tests))
        results = [result for group in grouped_results for result in group]
        results.sort(key=lambda result: result["index"])

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


class OpenCodeInstructEnv:
    name = "opencodeinstruct"

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
        memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
        max_test_workers: int = DEFAULT_MAX_TEST_WORKERS,
    ) -> None:
        # Goal / observation: implement this natural-language programming task.
        self.example_id: str | None = None
        self.problem: str | None = None

        # Hidden verifier data from the dataset row.
        self.unit_tests: list[str] = []
        self.total_assertions: int = 0

        # Test-process resource controls.
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.max_test_workers = max_test_workers

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

        self.parse_unit_tests = parse_unit_tests
        self.extract_python_code = extract_python_code
        self.check_code = check_code

    def build_prompt(self) -> str:
        """Turn the current problem into a model prompt without leaking tests or code."""
        return (
            "Write Python code that solves the following problem.\n\n"
            f"{self.problem}\n\n"
            "Put step-by-step reasoning in <think>...</think> and the complete "
            "executable Python solution in <answer>...</answer>. The answer may use "
            "a Python Markdown code fence, but must not contain prose outside it."
        )

    def state(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "input": self.problem,
            "current_step": self.current_step,
            "last_response": self.last_response,
            "terminated": self.terminated,
            "prompt": self.prompt,
        }

    def load_problem(
        self,
        problem: dict[str, Any] | str,
        unit_tests: str | Sequence[str] | None = None,
        example_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Bind one dataset row and reset all per-episode values.

        The preferred form matches ``make_env_reward_fn``::

            env.load_problem({
                "id": "...",
                "input": "...",
                "unit_tests": "[...]",
            })

        A positional form is also available for standalone callers::

            env.load_problem(problem, unit_tests, example_id)
        """
        if isinstance(problem, dict):
            if unit_tests is not None or example_id is not None:
                raise TypeError(
                    "when problem is a dict, do not pass unit_tests or example_id"
                )
            row = problem
            problem = row["input"]
            unit_tests = row["unit_tests"]
            example_id = row.get("id")

        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("input must be a non-empty string")
        if unit_tests is None:
            raise ValueError("unit_tests is required")
        tests = self.parse_unit_tests(unit_tests)

        self.example_id = None if example_id is None else str(example_id)
        self.problem = problem.strip()
        self.unit_tests = tests
        self.total_assertions = _count_assertions(tests)
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
            "id": self.example_id,
            "input": self.problem,
            "prompt": self.prompt,
        }

    def reset(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Start a new episode from one OpenCodeInstruct dataset row.

        1. Read id / input / unit_tests from the row.
        2. Parse and validate the hidden assertion snippets.
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
                "observation": {"id": self.example_id, "input": self.problem},
                "code": self.code,
                "reward": self.reward if reward is ... else reward,
                "acceptor": self.acceptor,
                "terminated": self.terminated,
                "severity": self.severity,
                "check": result,
                "state": self.state(),
            }

        # Episode lifecycle guards — check before mutating scored state.
        if self.problem is None or not self.unit_tests:
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
                self.unit_tests,
                timeout_seconds=self.timeout_seconds,
                memory_limit_mb=self.memory_limit_mb,
                max_test_workers=self.max_test_workers,
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
            if isinstance(exc, (MemoryError, SystemError)):
                self.severity = SEVERITY_STOP_TRAINING
            else:
                self.severity = SEVERITY_END_ROLLOUT
            return _out(result)  # reward stays None

        self.reward = float(result["reward"])
        self.acceptor = bool(result["correct"])
        self.severity = None if self.acceptor else SEVERITY_END_EPISODE
        return _out(result)
