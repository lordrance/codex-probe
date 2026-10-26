"""Verify a Phase 2 run, and compare a direct run with a proxied one.

In a Phase 2 run, the real ``codex exec`` talks to CodexProbe, which talks to the
mock Responses server. The mock scripts a small coding task and saves every
exchange to a capture directory. The verdict never rests on Codex's exit code
alone:

* the expected files must exist in the workspace;
* the workspace's tests must pass when they are run again here, with an
  interpreter that the caller names;
* the mock must have issued at least one ``exec_command`` call, and the
  following request must carry a ``function_call_output`` for every call issued;
* CodexProbe's log must hold one ``completed`` call for each ``POST /v1/responses``
  the mock answered, and each request and response body must equal, byte for
  byte, what the mock saved.

A direct run of the same scenario (Codex talks to the mock, with no proxy) can be
compared with the proxied run. Only content that stays the same from one run to
the next is compared:

* the requests' methods and paths;
* the types and roles of the ``input`` items;
* the tool names;
* the function calls that Codex sends back in each request's ``input`` (name,
  call_id, arguments), and the call_ids of their outputs;
* every regular file in the two workspaces, by relative path and raw bytes, as
  Codex left them (the snapshot is taken before the checker runs the tests);
* the test results.

Ids, times, paths and durations are not compared, and no rules are written to
normalise them.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codex_probe.mock_server import Exchange, load_capture
from codex_probe.storage import SessionLoadError, decode_body, load_session

TEST_TIMEOUT_S = 300


@dataclass(frozen=True)
class Check:
    """One named check and its result."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Report:
    """Every check of a verification. It passes only if there are checks and all of them passed."""

    checks: tuple[Check, ...]

    @property
    def passed(self) -> bool:
        """True if every check passed."""
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        """The checks that failed."""
        return [check for check in self.checks if not check.passed]

    def text(self) -> str:
        """One PASS or FAIL line per check, then the overall result."""
        lines = [f"{'PASS' if check.passed else 'FAIL'}  {check.name}: {check.detail}" for check in self.checks]
        lines.append(f"RESULT: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


@dataclass(frozen=True)
class RunEvidence:
    """Where one run left its evidence."""

    workspace: Path
    codex_exit_code: int
    capture_dir: Path  # the mock server's capture_dir
    run_dir: Path | None = None  # CodexProbe's run directory; None for a direct run


@dataclass(frozen=True)
class _RunView:
    """What was read from one run, kept for the comparison."""

    exchanges: list[Exchange] | None  # None if the capture could not be read
    files: dict[str, bytes | None]  # expected file -> its bytes, or None if missing
    tree: dict[str, bytes]  # every regular file in the workspace, by relative path, before the test run
    tests_passed: bool


def verify(
    proxied: RunEvidence,
    *,
    expected_files: Sequence[str],
    python: str,
    test_args: Sequence[str] = ("-m", "unittest"),
    direct: RunEvidence | None = None,
) -> Report:
    """Verify a proxied Phase 2 run. With ``direct``, also verify that run and compare the two.

    Args:
        proxied: the run through CodexProbe. Its ``run_dir`` is required.
        expected_files: files the task must produce, relative to each workspace.
        python: the interpreter that runs the workspace's tests again.
        test_args: the arguments for that interpreter. The default runs unittest discovery.
        direct: an optional run of the same scenario, with Codex talking to the mock directly.
    """
    if proxied.run_dir is None:
        raise ValueError("the proxied run needs CodexProbe's run_dir")
    if not expected_files:
        raise ValueError("expected_files must name at least one file")
    checks, proxied_view = _examine("proxied run", proxied, expected_files, python, test_args)
    checks += check_log(proxied.run_dir, proxied_view.exchanges)
    if direct is not None:
        direct_checks, direct_view = _examine("direct run", direct, expected_files, python, test_args)
        checks += direct_checks
        checks += _compare(direct_view, proxied_view)
    return Report(tuple(checks))


# --- one run -----------------------------------------------------------------


def _examine(
    label: str,
    run: RunEvidence,
    expected_files: Sequence[str],
    python: str,
    test_args: Sequence[str],
) -> tuple[list[Check], _RunView]:
    checks = [
        Check(f"{label}: codex exit code", run.codex_exit_code == 0, f"codex exec exited with {run.codex_exit_code}")
    ]
    files = {name: _read_file(run.workspace / name) for name in expected_files}
    missing = [name for name, data in files.items() if data is None]
    if missing:
        detail = f"missing from {run.workspace}: {', '.join(missing)}"
    else:
        detail = ", ".join(f"{name} ({len(data or b'')} bytes)" for name, data in files.items())
    checks.append(Check(f"{label}: expected files", not missing, detail))
    tree = _workspace_files(run.workspace)  # before the test run below adds its own files
    tests_passed, detail = run_workspace_tests(run.workspace, python, test_args)
    checks.append(Check(f"{label}: workspace tests", tests_passed, detail))
    exchanges, capture_check = check_capture(label, run.capture_dir)
    checks.append(capture_check)
    checks.append(check_tool_calls(label, exchanges))
    return checks, _RunView(exchanges, files, tree, tests_passed)


def _read_file(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def _workspace_files(workspace: Path) -> dict[str, bytes]:
    """Every regular file under the workspace, keyed by its relative path with ``/``."""
    if not workspace.is_dir():
        return {}
    return {path.relative_to(workspace).as_posix(): path.read_bytes() for path in workspace.rglob("*") if path.is_file()}


def run_workspace_tests(workspace: Path, python: str, test_args: Sequence[str] = ("-m", "unittest")) -> tuple[bool, str]:
    """Run the workspace's tests again with ``python``; return whether they passed, and a short detail."""
    command = [python, *test_args]
    if not workspace.is_dir():
        return False, f"workspace {workspace} does not exist"
    try:
        done = subprocess.run(
            command,
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TEST_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not run {' '.join(command)}: {exc}"
    lines = (done.stdout + done.stderr).strip().splitlines()
    return done.returncode == 0, f"{' '.join(command)} exited with {done.returncode}: {' | '.join(lines[-3:])}"


def check_capture(label: str, capture_dir: Path) -> tuple[list[Exchange] | None, Check]:
    """Read the mock's capture. It passes if the mock answered at least one
    ``POST /v1/responses`` and sent every response exactly as scripted.

    Returns the exchanges (None if the capture cannot be read) and the check.
    """
    name = f"{label}: mock capture"
    try:
        exchanges = load_capture(capture_dir)
    except (OSError, ValueError, KeyError) as exc:
        return None, Check(name, False, f"cannot read the capture in {capture_dir}: {type(exc).__name__}: {exc}")
    posts = _responses_exchanges(exchanges)
    problems = [] if posts else ["the mock received no POST /v1/responses"]
    problems += [f"exchange {e.index}: the mock refused it: {e.failure}" for e in exchanges if e.failure]
    problems += [f"exchange {e.index}: ended as {e.state!r}, not sent" for e in exchanges if e.state != "sent"]
    detail = "; ".join(problems) or (
        f"{len(exchanges)} exchange(s), {len(posts)} POST /v1/responses, all answered as scripted"
    )
    return exchanges, Check(name, not problems, detail)


def check_tool_calls(label: str, exchanges: list[Exchange] | None) -> Check:
    """Pass if the mock issued at least one ``exec_command`` call and the next
    request carried a ``function_call_output`` for every call the mock issued."""
    name = f"{label}: tool calls"
    if exchanges is None:
        return Check(name, False, "there is no capture to check")
    posts = _responses_exchanges(exchanges)
    issued = [(index, call) for index, exchange in enumerate(posts) for call in exchange.function_calls]
    if not any(call.name == "exec_command" for _, call in issued):
        return Check(name, False, "the mock never issued an exec_command call, so no tool ever ran")
    problems = []
    for index, call in issued:
        if index + 1 >= len(posts):
            problems.append(f"{call.call_id} ({call.name}): no later request carried its output")
        elif call.call_id not in _function_call_output_ids(posts[index + 1].request_body):
            problems.append(f"{call.call_id} ({call.name}): request {index + 1} has no function_call_output for it")
    detail = "; ".join(problems) or (
        f"{len(issued)} call(s), each answered by the next request: "
        + ", ".join(f"{call.call_id} ({call.name})" for _, call in issued)
    )
    return Check(name, not problems, detail)


# --- CodexProbe's log against the mock's capture -----------------------------


def check_log(run_dir: Path, exchanges: list[Exchange] | None) -> list[Check]:
    """Check CodexProbe's log in ``run_dir`` against the mock's exchanges.

    The first check passes if the log holds one ``completed`` call, numbered in
    order, for each ``POST /v1/responses`` the mock answered. The second passes
    if every logged request and response body equals the mock's bytes.
    """
    log_name = "proxied run: CodexProbe log"
    bytes_name = "proxied run: logged bodies equal the mock's bytes"
    try:
        calls = load_session(run_dir)
    except SessionLoadError as exc:
        return [Check(log_name, False, str(exc)), Check(bytes_name, False, "there is no log to compare")]
    problems = []
    seqs = [call["seq"] for call in calls]
    if seqs != list(range(len(calls))):
        problems.append(f"seq should run 0..{len(calls) - 1} without gaps, got {seqs}")
    problems += [f"seq {call['seq']} ended as {call['outcome']!r}" for call in calls if call["outcome"] != "completed"]
    posts = _responses_exchanges(exchanges) if exchanges is not None else None
    if posts is not None and len(calls) != len(posts):
        problems.append(f"{len(calls)} call(s) logged, but the mock answered {len(posts)} POST /v1/responses")
    if not calls:
        problems.append("the log holds no calls")
    detail = "; ".join(problems) or f"{len(calls)} call(s), seq 0..{len(calls) - 1}, all completed"
    checks = [Check(log_name, not problems, detail)]
    if posts is None:
        checks.append(Check(bytes_name, False, "there is no capture to compare with"))
        return checks
    mismatches = []
    for index, (call, exchange) in enumerate(zip(calls, posts)):
        for part, logged, seen in (
            ("request", call["request"]["body"], exchange.request_body),
            ("response", call["response"]["body"], exchange.response_body),
        ):
            logged_bytes = decode_body(logged) if logged is not None else b""
            if logged_bytes != seen:
                mismatches.append(f"call {index} {part} body: {_difference(logged_bytes, seen)}")
    if len(calls) != len(posts):
        mismatches.append(f"{len(calls)} logged call(s) against {len(posts)} captured")
    detail = "; ".join(mismatches) or f"all {len(calls)} request and response bodies are identical, byte for byte"
    checks.append(Check(bytes_name, not mismatches, detail))
    return checks


def _difference(logged: bytes, seen: bytes) -> str:
    for offset, (a, b) in enumerate(zip(logged, seen)):
        if a != b:
            return f"{len(logged)} bytes logged, {len(seen)} captured, first difference at byte {offset}"
    return f"{len(logged)} bytes logged, {len(seen)} captured, one is a prefix of the other"


# --- direct against proxied --------------------------------------------------


def _compare(direct: _RunView, proxied: _RunView) -> list[Check]:
    label = "direct vs proxied"
    if direct.exchanges is None or proxied.exchanges is None:
        return [Check(f"{label}: captures", False, "a capture could not be read, so the runs cannot be compared")]
    direct_posts = _responses_exchanges(direct.exchanges)
    proxied_posts = _responses_exchanges(proxied.exchanges)
    checks = [
        _same(
            f"{label}: requests (method and path)",
            [(e.method, e.path) for e in direct.exchanges],
            [(e.method, e.path) for e in proxied.exchanges],
        ),
        _same(
            f"{label}: input item types and roles",
            [_input_shape(e) for e in direct_posts],
            [_input_shape(e) for e in proxied_posts],
        ),
        _same(
            f"{label}: tool names",
            [_tool_names(e) for e in direct_posts],
            [_tool_names(e) for e in proxied_posts],
        ),
        _same(
            f"{label}: function calls (name, call_id, arguments)",
            [_function_call_items(e) for e in direct_posts],
            [_function_call_items(e) for e in proxied_posts],
        ),
        _same(
            f"{label}: call_ids of function_call_output items",
            [_function_call_output_ids(e.request_body) for e in direct_posts],
            [_function_call_output_ids(e.request_body) for e in proxied_posts],
        ),
    ]
    missing = [f"{which}: {name}" for which, view in (("direct", direct), ("proxied", proxied))
               for name, data in view.files.items() if data is None]
    if missing:
        checks.append(Check(f"{label}: final workspace files", False, f"missing: {', '.join(missing)}"))
    else:
        problems = []
        for which, mine, other in (("direct", direct.tree, proxied.tree), ("proxied", proxied.tree, direct.tree)):
            extra = sorted(mine.keys() - other.keys())
            if extra:
                problems.append(f"only in {which}: {', '.join(extra)}")
        different = sorted(name for name in direct.tree.keys() & proxied.tree.keys() if direct.tree[name] != proxied.tree[name])
        if different:
            problems.append(f"differ: {', '.join(different)}")
        detail = "; ".join(problems) or f"identical: {len(direct.tree)} file(s): {', '.join(sorted(direct.tree))}"
        checks.append(Check(f"{label}: final workspace files", not problems, detail))
    checks.append(
        Check(
            f"{label}: workspace tests",
            direct.tests_passed and proxied.tests_passed,
            f"direct {'passed' if direct.tests_passed else 'failed'}, "
            f"proxied {'passed' if proxied.tests_passed else 'failed'}",
        )
    )
    return checks


def _same(name: str, direct: Sequence[Any], proxied: Sequence[Any]) -> Check:
    if list(direct) == list(proxied):
        return Check(name, True, f"identical ({len(direct)} item(s))")
    if len(direct) != len(proxied):
        return Check(name, False, f"direct has {len(direct)} item(s), proxied has {len(proxied)}")
    index = next(i for i, (a, b) in enumerate(zip(direct, proxied)) if a != b)
    return Check(name, False, f"differ at request {index}: direct {_short(direct[index])} vs proxied {_short(proxied[index])}")


def _short(value: Any, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --- reading request bodies --------------------------------------------------


def _responses_exchanges(exchanges: Sequence[Exchange]) -> list[Exchange]:
    return [e for e in exchanges if e.method == "POST" and e.path == "/v1/responses"]


def _request_json(body: bytes) -> Mapping[str, Any]:
    try:
        data = json.loads(body)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _items(body: bytes, key: str) -> list[Mapping[str, Any]]:
    value = _request_json(body).get(key)
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _input_shape(exchange: Exchange) -> list[tuple[Any, Any]]:
    """The (type, role) of each input item, e.g. ("message", "developer"), ("function_call", None)."""
    return [(item.get("type"), item.get("role")) for item in _items(exchange.request_body, "input")]


def _tool_names(exchange: Exchange) -> list[Any]:
    """Each tool's name, or its type for tools without a name (such as a hosted web_search)."""
    return [tool.get("name") or tool.get("type") for tool in _items(exchange.request_body, "tools")]


def _function_call_items(exchange: Exchange) -> list[tuple[Any, Any, Any]]:
    """The function calls Codex sent back in this request's ``input``: (name, call_id, arguments).

    These are read from the request the mock actually received. The mock's own
    scripted calls (``Exchange.function_calls``) are the same in every run of a
    script, so comparing those would prove nothing.
    """
    return [
        (item.get("name"), item.get("call_id"), item.get("arguments"))
        for item in _items(exchange.request_body, "input")
        if item.get("type") == "function_call"
    ]


def _function_call_output_ids(body: bytes) -> list[Any]:
    return [item.get("call_id") for item in _items(body, "input") if item.get("type") == "function_call_output"]
