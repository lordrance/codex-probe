"""Tests for codex_probe.e2e_validation and scripts/verify_e2e.py.

The evidence here comes from the real mock server and the real ProxyRecorder,
but a small Python client plays Codex's part. It is synthetic evidence made to
test the checker. It is not an end-to-end result: real Codex is not involved.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from codex_probe import ProxyRecorder
from codex_probe.e2e_validation import Check, Report, RunEvidence, verify
from codex_probe.mock_server import (
    MockResponse,
    MockResponsesServer,
    ev_apply_patch_exec_command_call,
    ev_assistant_message,
    ev_completed,
    ev_exec_command_call,
    ev_response_created,
    load_capture,
    sse_response,
)
from codex_probe.streaming import ResponseStreamAssembler

PYTHON = sys.executable
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_e2e.py"
EXPECTED = ["calc.py", "test_calc.py"]
FILES = {
    "calc.py": "def add(a, b):\n    return a + b\n",
    "test_calc.py": (
        "import unittest\n\nfrom calc import add\n\n\n"
        "class AddTest(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n"
    ),
}
PATCH = "*** Begin Patch\n" + "".join(
    f"*** Add File: {name}\n" + "".join(f"+{line}\n" for line in text.splitlines()) for name, text in FILES.items()
) + "*** End Patch"
TOOLS = [
    {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
    {"type": "function", "name": "write_stdin", "parameters": {"type": "object"}},
]
CHECK_NAMES = [
    "proxied run: codex exit code",
    "proxied run: expected files",
    "proxied run: workspace tests",
    "proxied run: mock capture",
    "proxied run: tool calls",
    "proxied run: CodexProbe log",
    "proxied run: logged bodies equal the mock's bytes",
    "direct run: codex exit code",
    "direct run: expected files",
    "direct run: workspace tests",
    "direct run: mock capture",
    "direct run: tool calls",
    "direct vs proxied: requests (method and path)",
    "direct vs proxied: input item types and roles",
    "direct vs proxied: tool names",
    "direct vs proxied: function calls (name, call_id, arguments)",
    "direct vs proxied: call_ids of function_call_output items",
    "direct vs proxied: final workspace files",
    "direct vs proxied: workspace tests",
]


def coding_task() -> list[MockResponse]:
    """The Phase 2 script: write the files, run the tests, give the final answer."""
    return [
        sse_response([ev_response_created("resp_1"), ev_apply_patch_exec_command_call("call_1", PATCH), ev_completed("resp_1")]),
        sse_response([ev_response_created("resp_2"), ev_exec_command_call("call_2", "python -m unittest"), ev_completed("resp_2")]),
        sse_response(
            [ev_response_created("resp_3"), ev_assistant_message("msg_1", "calc.py and its test pass."), ev_completed("resp_3")]
        ),
    ]


def claims_done_without_tools() -> list[MockResponse]:
    """A model that only says it is done."""
    return [sse_response([ev_response_created("resp_1"), ev_assistant_message("msg_1", "完成了"), ev_completed("resp_1")])]


def play_codex(
    base_url: str,
    workspace: Path,
    *,
    extra_message: bool = False,
    skip_outputs: bool = False,
    reformat_echoed_arguments: bool = False,
) -> int:
    """Stand in for Codex: send each turn, apply the patch, and report the tools' output.

    Each run uses a fresh session id, so request bodies differ between runs, just
    as they do with real Codex. Returns the exit code Codex would have returned.
    ``reformat_echoed_arguments`` echoes each function call back into the history
    with its arguments re-serialised: the same JSON, but different bytes.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    session = str(uuid.uuid4())
    history: list[dict[str, Any]] = [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "You are a coding agent."}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Write add() and a test."}]},
    ]
    if extra_message:
        history.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hurry."}]})
    with httpx.Client(trust_env=False, timeout=30) as client:
        for _ in range(5):
            body = {"model": "mock-model", "input": history, "tools": TOOLS, "stream": True,
                    "client_metadata": {"session_id": session}}
            response = client.post(
                f"{base_url}/responses",
                content=json.dumps(body).encode(),
                headers={"content-type": "application/json", "session-id": session},
            )
            if response.status_code != 200:
                return 1
            assembler = ResponseStreamAssembler()
            assembler.feed(response.content)
            items = assembler.finish().items
            for item in items:
                if reformat_echoed_arguments and item.get("type") == "function_call":
                    item = {**item, "arguments": json.dumps(json.loads(item["arguments"]))}
                history.append(item)
            calls = [item for item in items if item.get("type") == "function_call"]
            if not calls:
                return 0  # a final answer ends the turn
            for call in calls:
                if json.loads(call["arguments"])["cmd"].startswith("apply_patch"):
                    for name, text in FILES.items():
                        (workspace / name).write_text(text, encoding="utf-8")
                    output = "Success. Updated the following files:\nA calc.py\nA test_calc.py\n"
                else:
                    output = "Process exited with code 0\nOutput:\n.\nOK\n"
                if not skip_outputs:
                    history.append({"type": "function_call_output", "call_id": call["call_id"], "output": output})
    return 1


def record_run(root: Path, script: list[MockResponse], *, proxied: bool, **play: bool) -> RunEvidence:
    """Run the mock (and CodexProbe if ``proxied``) and play Codex's part; return the evidence."""
    capture = root / "capture"
    mock = MockResponsesServer(script, capture_dir=capture)
    recorder: ProxyRecorder | None = None
    try:
        mock.start()
        assert mock.base_url is not None
        base_url = mock.base_url
        if proxied:
            recorder = ProxyRecorder(
                {"upstream": {"base_url": mock.base_url, "auth": {"mode": "none"}}, "log_dir": str(root / "logs")}
            )
            base_url = recorder.start()
        exit_code = play_codex(base_url, root / "workspace", **play)
        mock.wait_for_exchanges(len(mock.exchanges))  # every capture is on disk
    finally:
        if recorder is not None:
            recorder.stop()
        mock.stop()
    return RunEvidence(root / "workspace", exit_code, capture, recorder.run_dir if recorder is not None else None)


@pytest.fixture(scope="module")
def golden(tmp_path_factory: pytest.TempPathFactory) -> dict[str, RunEvidence]:
    root = tmp_path_factory.mktemp("golden")
    return {
        "proxied": record_run(root / "proxied", coding_task(), proxied=True),
        "direct": record_run(root / "direct", coding_task(), proxied=False),
    }


@pytest.fixture
def evidence(golden: dict[str, RunEvidence], tmp_path: Path) -> dict[str, RunEvidence]:
    """A private copy of the golden evidence that a test may damage."""
    copies = {}
    for name, run in golden.items():
        root = tmp_path / name
        copies[name] = RunEvidence(
            Path(shutil.copytree(run.workspace, root / "workspace")),
            run.codex_exit_code,
            Path(shutil.copytree(run.capture_dir, root / "capture")),
            Path(shutil.copytree(run.run_dir, root / "run")) if run.run_dir is not None else None,
        )
    return copies


def failed(report: Report) -> list[str]:
    return [check.name for check in report.failures]


def find(report: Report, name: str) -> Check:
    return next(check for check in report.checks if check.name == name)


def session_log(run: RunEvidence) -> Path:
    assert run.run_dir is not None
    [path] = run.run_dir.glob("session-*.jsonl")
    return path


# --- a genuine run passes ------------------------------------------------------


def test_a_genuine_run_passes_every_check(evidence: dict[str, RunEvidence]) -> None:
    report = verify(evidence["proxied"], expected_files=EXPECTED, python=PYTHON, direct=evidence["direct"])
    assert report.passed, report.text()
    assert [check.name for check in report.checks] == CHECK_NAMES
    assert find(report, "proxied run: CodexProbe log").detail == "3 call(s), seq 0..2, all completed"
    assert find(report, "proxied run: tool calls").detail == (
        "2 call(s), each answered by the next request: call_1 (exec_command), call_2 (exec_command)"
    )
    assert find(report, "proxied run: logged bodies equal the mock's bytes").detail == (
        "all 3 request and response bodies are identical, byte for byte"
    )
    # The raw bodies of the two runs differ (session ids), yet no normalisation was needed.
    direct_first = load_capture(evidence["direct"].capture_dir)[0].request_body
    proxied_first = load_capture(evidence["proxied"].capture_dir)[0].request_body
    assert direct_first != proxied_first
    # Workspaces are compared as the run left them: the checker's own test run
    # writes __pycache__, but that is after the snapshot.
    assert find(report, "direct vs proxied: final workspace files").detail == (
        "identical: 2 file(s): calc.py, test_calc.py"
    )
    if not sys.dont_write_bytecode:
        assert (evidence["proxied"].workspace / "__pycache__").is_dir()
        assert (evidence["direct"].workspace / "__pycache__").is_dir()
    assert report.text().endswith("RESULT: PASS")


# --- the three negative controls the requirements name ----------------------------


def test_exit_code_zero_but_missing_file_fails(evidence: dict[str, RunEvidence]) -> None:
    (evidence["proxied"].workspace / "calc.py").unlink()
    report = verify(evidence["proxied"], expected_files=EXPECTED, python=PYTHON, direct=evidence["direct"])
    assert not report.passed
    assert find(report, "proxied run: codex exit code").passed  # exit code 0 alone proves nothing
    assert failed(report) == [
        "proxied run: expected files",
        "proxied run: workspace tests",
        "direct vs proxied: final workspace files",
        "direct vs proxied: workspace tests",
    ]
    assert find(report, "proxied run: expected files").detail.endswith(": calc.py")
    assert find(report, "direct vs proxied: final workspace files").detail == "missing: proxied: calc.py"


def test_saying_done_without_any_tool_call_fails(tmp_path: Path) -> None:
    run = record_run(tmp_path, claims_done_without_tools(), proxied=True)
    assert run.codex_exit_code == 0
    report = verify(run, expected_files=EXPECTED, python=PYTHON)
    assert failed(report) == ["proxied run: expected files", "proxied run: workspace tests", "proxied run: tool calls"]
    assert find(report, "proxied run: tool calls").detail == "the mock never issued an exec_command call, so no tool ever ran"
    # The log itself is fine: the failure is the task, not the recording.
    assert find(report, "proxied run: CodexProbe log").passed
    assert find(report, "proxied run: logged bodies equal the mock's bytes").passed


def _drop_last_log_line(run: RunEvidence) -> None:
    path = session_log(run)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]))


def _flip_a_byte(path: Path, offset: int = 12) -> None:
    """Change one byte. Offset 12 of a request is the "o" of "mock-model", so the JSON stays valid."""
    data = bytearray(path.read_bytes())
    data[offset] ^= 0x01
    path.write_bytes(bytes(data))


DAMAGE: list[tuple[str, Callable[[RunEvidence], Any], list[str], str]] = [
    (
        "session log deleted",
        lambda run: session_log(run).unlink(),
        ["proxied run: CodexProbe log", "proxied run: logged bodies equal the mock's bytes"],
        "0 call(s) logged, but the mock answered 3 POST /v1/responses",
    ),
    (
        "session log cut off",
        lambda run: session_log(run).write_bytes(session_log(run).read_bytes()[:-40]),
        ["proxied run: CodexProbe log", "proxied run: logged bodies equal the mock's bytes"],
        "last line has no newline",
    ),
    (
        "one call missing from the log",
        _drop_last_log_line,
        ["proxied run: CodexProbe log", "proxied run: logged bodies equal the mock's bytes"],
        "2 call(s) logged, but the mock answered 3 POST /v1/responses",
    ),
    (
        "request bytes changed in the capture",
        lambda run: _flip_a_byte(run.capture_dir / "0001-request.bin"),
        ["proxied run: logged bodies equal the mock's bytes"],
        "call 1 request body:",
    ),
    (
        "response bytes changed in the capture",
        lambda run: _flip_a_byte(run.capture_dir / "0002-response.bin", offset=30),
        ["proxied run: logged bodies equal the mock's bytes"],
        "call 2 response body:",
    ),
    (
        "response file missing from the capture",
        lambda run: (run.capture_dir / "0002-response.bin").unlink(),
        [
            "proxied run: mock capture",
            "proxied run: tool calls",
            "proxied run: logged bodies equal the mock's bytes",
        ],
        "cannot read the capture",
    ),
    (
        "response file cut short in the capture",
        lambda run: (run.capture_dir / "0000-response.bin").write_bytes(
            (run.capture_dir / "0000-response.bin").read_bytes()[:-5]
        ),
        [
            "proxied run: mock capture",
            "proxied run: tool calls",
            "proxied run: logged bodies equal the mock's bytes",
        ],
        "0000-response.bin has",
    ),
]


@pytest.mark.parametrize(("damage", "failing", "message"), [pytest.param(d, f, m, id=i) for i, d, f, m in DAMAGE])
def test_missing_or_altered_log_or_capture_fails(
    evidence: dict[str, RunEvidence], damage: Callable[[RunEvidence], Any], failing: list[str], message: str
) -> None:
    damage(evidence["proxied"])
    report = verify(evidence["proxied"], expected_files=EXPECTED, python=PYTHON)
    assert failed(report) == failing, report.text()
    assert message in "\n".join(check.detail for check in report.failures)


# --- other failures ----------------------------------------------------------


def test_direct_run_with_different_stable_content_fails(evidence: dict[str, RunEvidence], tmp_path: Path) -> None:
    direct = record_run(tmp_path / "direct-variant", coding_task(), proxied=False, extra_message=True)
    report = verify(evidence["proxied"], expected_files=EXPECTED, python=PYTHON, direct=direct)
    assert failed(report) == ["direct vs proxied: input item types and roles"]
    assert find(report, "direct vs proxied: input item types and roles").detail.startswith("differ at request 0:")


def test_direct_run_that_echoes_altered_function_calls_fails(evidence: dict[str, RunEvidence], tmp_path: Path) -> None:
    direct = record_run(tmp_path / "direct-altered", coding_task(), proxied=False, reformat_echoed_arguments=True)
    report = verify(evidence["proxied"], expected_files=EXPECTED, python=PYTHON, direct=direct)
    assert failed(report) == ["direct vs proxied: function calls (name, call_id, arguments)"], report.text()
    detail = find(report, "direct vs proxied: function calls (name, call_id, arguments)").detail
    assert detail.startswith("differ at request 1:")
    assert '{"cmd": "apply_patch' in detail and '{"cmd":"apply_patch' in detail
    # The mock issued exactly the same scripted calls in both runs, so its own record
    # of them could not show the difference. Only the requests it received can.
    direct_issued = [exchange.function_calls for exchange in load_capture(direct.capture_dir)]
    proxied_issued = [exchange.function_calls for exchange in load_capture(evidence["proxied"].capture_dir)]
    assert direct_issued == proxied_issued


@pytest.mark.parametrize("side", ["direct", "proxied"])
def test_an_extra_file_in_either_workspace_fails(evidence: dict[str, RunEvidence], side: str) -> None:
    extra = evidence[side].workspace / "notes" / "extra.txt"  # nested: the comparison walks every folder
    extra.parent.mkdir()
    extra.write_bytes(b"not part of the task\n")
    report = verify(evidence["proxied"], expected_files=EXPECTED, python=PYTHON, direct=evidence["direct"])
    assert failed(report) == ["direct vs proxied: final workspace files"], report.text()
    assert find(report, "direct vs proxied: final workspace files").detail == f"only in {side}: notes/extra.txt"


def test_failing_workspace_tests_fail(evidence: dict[str, RunEvidence]) -> None:
    test_file = evidence["proxied"].workspace / "test_calc.py"
    test_file.write_text(test_file.read_text("utf-8").replace("5)", "6)"), encoding="utf-8")
    report = verify(evidence["proxied"], expected_files=EXPECTED, python=PYTHON, direct=evidence["direct"])
    assert failed(report) == [
        "proxied run: workspace tests",
        "direct vs proxied: final workspace files",
        "direct vs proxied: workspace tests",
    ]
    assert "exited with 1" in find(report, "proxied run: workspace tests").detail
    assert find(report, "direct vs proxied: final workspace files").detail == "differ: test_calc.py"


def test_a_request_without_the_tool_output_fails(tmp_path: Path) -> None:
    run = record_run(tmp_path, coding_task(), proxied=True, skip_outputs=True)
    report = verify(run, expected_files=EXPECTED, python=PYTHON)
    assert run.codex_exit_code == 1
    assert {"proxied run: codex exit code", "proxied run: mock capture", "proxied run: tool calls"} <= set(failed(report))
    assert "the mock refused it" in find(report, "proxied run: mock capture").detail
    assert "request 1 has no function_call_output for it" in find(report, "proxied run: tool calls").detail


def test_verify_rejects_unusable_arguments(evidence: dict[str, RunEvidence]) -> None:
    with pytest.raises(ValueError, match="needs CodexProbe's run_dir"):
        verify(evidence["direct"], expected_files=EXPECTED, python=PYTHON)
    with pytest.raises(ValueError, match="at least one file"):
        verify(evidence["proxied"], expected_files=[], python=PYTHON)


# --- the command-line script ---------------------------------------------------


def run_script(run: RunEvidence, *extra: str) -> subprocess.CompletedProcess[str]:
    assert run.run_dir is not None
    command = [
        PYTHON, str(SCRIPT),
        "--workspace", str(run.workspace), "--exit-code", str(run.codex_exit_code),
        "--capture", str(run.capture_dir), "--run-dir", str(run.run_dir),
        "--python", PYTHON, "--expect", "calc.py", "--expect", "test_calc.py", *extra,
    ]
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=300)


def test_script_exit_codes(evidence: dict[str, RunEvidence]) -> None:
    direct = evidence["direct"]
    good = run_script(
        evidence["proxied"],
        "--direct-workspace", str(direct.workspace),
        "--direct-exit-code", str(direct.codex_exit_code),
        "--direct-capture", str(direct.capture_dir),
    )
    assert good.returncode == 0, good.stdout + good.stderr
    assert good.stdout.strip().endswith("RESULT: PASS")
    assert good.stdout.count("PASS  ") == len(CHECK_NAMES)
    (evidence["proxied"].workspace / "calc.py").unlink()
    bad = run_script(evidence["proxied"])
    assert bad.returncode == 1
    assert "FAIL  proxied run: expected files" in bad.stdout and bad.stdout.strip().endswith("RESULT: FAIL")
    incomplete = run_script(evidence["proxied"], "--direct-workspace", str(direct.workspace))
    assert incomplete.returncode == 2 and "must be given together" in incomplete.stderr
