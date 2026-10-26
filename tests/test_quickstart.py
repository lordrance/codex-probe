"""Tests for examples/quickstart.py and the helpers it shares with the demo script.

The default suite never starts the real Codex CLI. The one end-to-end test is
marked ``e2e``, which the default options (``-m 'not e2e'``) leave out. Run it
with the real Codex 0.164.0:

    set CODEXPROBE_CODEX=<path to codex>
    set CODEXPROBE_CODEX_HOME=<a CODEX_HOME CodexProbe may write>   (optional)
    set CODEXPROBE_LOG_DIR=<where the run's files go>               (optional)
    python -m pytest -m e2e tests/test_quickstart.py
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import socket
import sys
import threading
import tomllib
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest

from codex_probe import ProxyRecorder
from codex_probe.codex_run import CONFIG_MARKER, CodexHomeError, codex_config, write_codex_config
from codex_probe.mock_server import (
    CODING_TASK_ANSWER,
    MockResponsesServer,
    coding_task,
    ev_assistant_message,
    ev_completed,
    ev_response_created,
    sse,
    sse_response,
    unittest_command,
)

PROJECT = Path(__file__).resolve().parents[1]


def load_quickstart() -> ModuleType:
    """examples/ is not a package, so load the script by its path."""
    spec = importlib.util.spec_from_file_location("quickstart", PROJECT / "examples" / "quickstart.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quickstart = load_quickstart()


# --- config.toml -------------------------------------------------------------


def test_codex_config_sends_codex_to_the_endpoint_and_turns_the_extras_off() -> None:
    text = codex_config("http://127.0.0.1:8135/v1")
    assert text.startswith(CONFIG_MARKER)
    config = tomllib.loads(text)
    assert config["model_provider"] == "codexprobe"
    assert config["model_providers"]["codexprobe"] == {
        "name": "CodexProbe",
        "base_url": "http://127.0.0.1:8135/v1",
        "wire_api": "responses",
        "request_max_retries": 0,
        "stream_max_retries": 0,
    }
    assert config["web_search"] == "disabled"
    assert config["features"] == {"unbounded_connection_retries": False, "plugins": False, "apps": False}
    assert config["analytics"] == {"enabled": False}
    assert config["windows"] == {"sandbox": "unelevated"}


def test_codex_config_for_a_local_model_sets_context_timeout_and_minimal_tools() -> None:
    config = tomllib.loads(
        codex_config(
            "http://127.0.0.1:8135/v1",
            model="qwen-codex",
            context_window=16384,
            stream_idle_timeout_ms=600000,
            minimal_tools=True,
        )
    )
    assert config["model"] == "qwen-codex"
    assert config["model_context_window"] == 16384
    assert config["model_reasoning_summary"] == "none"
    assert config["model_providers"]["codexprobe"]["stream_idle_timeout_ms"] == 600000
    assert config["features"] == {
        "unbounded_connection_retries": False,
        "plugins": False,
        "apps": False,
        "view_image": False,
        "tool_suggest": False,
        "goals": False,
    }
    assert config["agents"] == {"enabled": False}
    assert config["tools"] == {"experimental_request_user_input": {"enabled": False}}
    assert config["skills"] == {"include_instructions": False}
    assert config["windows"] == {"sandbox": "unelevated"}


def test_write_codex_config_replaces_only_its_own_file(tmp_path: Path) -> None:
    home = tmp_path / "home"
    path = write_codex_config(home, codex_config("http://127.0.0.1:1/v1"))  # creates the folder
    write_codex_config(home, codex_config("http://127.0.0.1:2/v1"))  # its own file: replaced
    assert "127.0.0.1:2/v1" in path.read_text("utf-8")
    path.write_text('model = "the user\'s own"\n', "utf-8")
    with pytest.raises(CodexHomeError, match="not written by CodexProbe"):
        write_codex_config(home, codex_config("http://127.0.0.1:3/v1"))
    assert path.read_text("utf-8") == 'model = "the user\'s own"\n'


def test_write_codex_config_refuses_the_default_codex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(CodexHomeError, match="default CODEX_HOME"):
        write_codex_config(tmp_path / ".codex", codex_config("http://127.0.0.1:1/v1"))
    assert not (tmp_path / ".codex").exists()


# --- the coding task -----------------------------------------------------------


def test_unittest_command_quotes_the_interpreter_for_each_shell() -> None:
    assert unittest_command(r"C:\Python312\python.exe", windows=True) == r"& 'C:\Python312\python.exe' -B -m unittest -v"
    assert unittest_command(r"C:\it's\python.exe", windows=True) == r"& 'C:\it''s\python.exe' -B -m unittest -v"
    assert unittest_command("/opt/my python/bin/python3", windows=False) == "'/opt/my python/bin/python3' -B -m unittest -v"


def test_coding_task_writes_the_files_runs_the_tests_then_answers() -> None:
    first, second, third = coding_task(r"C:\Python312\python.exe")
    patch_call, test_call = first.function_calls + second.function_calls
    assert json.loads(patch_call.arguments)["cmd"].startswith("apply_patch <<'EOF'\n*** Begin Patch\n*** Add File: calc.py")
    assert "*** Add File: test_calc.py" in json.loads(patch_call.arguments)["cmd"]
    assert json.loads(test_call.arguments) == {"cmd": unittest_command(r"C:\Python312\python.exe"), "yield_time_ms": 30000}
    assert third.function_calls == ()
    assert CODING_TASK_ANSWER.encode() in b"".join(third.chunks)
    assert [response.chunks[-1] for response in (first, second, third)] == [
        sse([ev_completed(f"resp-{n}")]) for n in (1, 2, 3)
    ]


# --- the quickstart's command line, prerequisites and summary ------------------


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([], id="no mode"),
        pytest.param(["--mode", "anthropic"], id="an unknown mode"),
        pytest.param(["--mode", "mock", "--timeout", "0"], id="zero timeout"),
        pytest.param(["--mode", "mock", "--timeout", "soon"], id="timeout not a number"),
        pytest.param(["--mode", "mock", "--task", "text"], id="a text task for the mock"),
    ],
)
def test_bad_arguments_exit_with_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        quickstart.main(argv)
    assert exc.value.code == 2


def test_default_log_dir_is_ignored_by_git() -> None:
    assert quickstart.DEFAULT_LOG_DIR == PROJECT / "runs"
    assert "runs/" in (PROJECT / ".gitignore").read_text("utf-8").splitlines()


def test_a_missing_codex_is_blocked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = quickstart.main(["--mode", "mock", "--codex", str(tmp_path / "no-codex.exe"), "--log-dir", str(tmp_path / "logs")])
    assert code == quickstart.BLOCKED == 3
    assert capsys.readouterr().out.startswith("BLOCKED: the Codex CLI was not found")
    assert not (tmp_path / "logs").exists()  # nothing ran


def test_a_codex_of_the_wrong_version_is_blocked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Python stands in for a wrong executable: its --version is not "codex-cli 0.164.0".
    argv = ["--mode", "mock", "--codex", sys.executable, "--codex-home", str(tmp_path / "home"), "--log-dir", str(tmp_path / "logs")]
    assert quickstart.main(argv) == 3
    out = capsys.readouterr().out
    assert out.startswith("BLOCKED: ") and "'Python 3." in out and "needs 'codex-cli 0.164.0'" in out
    assert not (tmp_path / "logs").exists()


def test_a_codex_home_with_a_foreign_config_is_blocked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text('model = "the user\'s own"\n', "utf-8")
    argv = ["--mode", "mock", "--codex", sys.executable, "--codex-home", str(home), "--log-dir", str(tmp_path / "logs")]
    assert quickstart.main(argv) == 3
    assert "not written by CodexProbe" in capsys.readouterr().out
    assert (home / "config.toml").read_text("utf-8") == 'model = "the user\'s own"\n'


def _call(seq: int, outcome: str, tool_calls: list[dict[str, Any]], first_byte: float | None) -> dict[str, Any]:
    return {
        "seq": seq,
        "outcome": outcome,
        "request": {"body": {"text": "{}", "json": {"model": "codexprobe-mock", "input": []}}},
        "derived": {"tool_calls": tool_calls},
        "timing": {"first_byte_ms": first_byte, "total_ms": None if first_byte is None else first_byte + 4.25},
    }


def test_summary_has_one_line_per_call() -> None:
    calls = [
        _call(0, "completed", [{"type": "function_call", "name": "exec_command", "call_id": "call-1"}], 36.5),
        _call(1, "stream_cut", [], None),
    ]
    assert quickstart.summarize(calls) == [
        "seq 0  model codexprobe-mock  completed  tool calls: exec_command (call-1)  first byte 36.5 ms, total 40.8 ms",
        "seq 1  model codexprobe-mock  stream_cut  tool calls: none  first byte n/a, total n/a",
    ]


# --- qwen mode: arguments, config, prerequisites, judging the reply ---------------


def test_qwen_mode_defaults() -> None:
    args = quickstart.parse_args(["--mode", "qwen"])
    assert args.ollama_url == "http://127.0.0.1:11434/v1"
    assert args.task == "code"  # the coding task; --task text is the smoke test
    assert quickstart.parse_args(["--mode", "qwen", "--qwen-task", "text"]).task == "text"  # the option's first name
    with pytest.raises(SystemExit) as exc:
        quickstart.parse_args(["--mode", "qwen", "--task", "chat"])
    assert exc.value.code == 2
    assert args.timeout is None and quickstart.QWEN_TIMEOUT_S == 3600  # main() falls back to 60 minutes
    assert quickstart.parse_args(["--mode", "qwen", "--ollama-url", "http://localhost:11434/v1/"]).ollama_url == (
        "http://localhost:11434/v1"
    )


@pytest.mark.parametrize(
    "url",
    [
        "ftp://127.0.0.1:11434/v1",
        "127.0.0.1:11434/v1",
        "http://127.0.0.1:11434",
        "http://127.0.0.1:11434/api",
        "http://user:secret@127.0.0.1:11434/v1",
        "http://127.0.0.1:11434/v1?x=1",
    ],
)
def test_a_bad_ollama_url_exits_with_2(url: str) -> None:
    with pytest.raises(SystemExit) as exc:
        quickstart.parse_args(["--mode", "qwen", "--ollama-url", url])
    assert exc.value.code == 2


def test_qwen_codex_config_matches_the_modelfile() -> None:
    config = tomllib.loads(quickstart.qwen_codex_config("http://127.0.0.1:8135/v1"))
    assert config["model"] == "codexprobe-qwen2.5-coder:1.5b"
    assert config["model_context_window"] == 16384
    provider = config["model_providers"]["codexprobe"]
    assert provider["name"] == "CodexProbe"  # "OpenAI" would switch Codex to OpenAI-only behaviour
    assert provider["base_url"] == "http://127.0.0.1:8135/v1"
    assert provider["stream_idle_timeout_ms"] == 600000
    assert provider["request_max_retries"] == provider["stream_max_retries"] == 0
    assert config["web_search"] == "disabled"
    assert config["analytics"] == {"enabled": False}
    assert config["features"]["plugins"] is False and config["features"]["apps"] is False
    assert config["agents"] == {"enabled": False}  # the minimal tool set
    assert config["windows"] == {"sandbox": "unelevated"}
    modelfile = (PROJECT / "examples" / "ollama" / "Modelfile").read_text("utf-8")
    assert f"PARAMETER num_ctx {quickstart.QWEN_CONTEXT}" in modelfile  # Codex and Ollama agree on the context


class _FakeOllama(BaseHTTPRequestHandler):
    """Answers /api/version and /api/show the way Ollama does, as set on the server.

    ``version_reply`` and ``show_reply`` are the raw bodies of the two 200 replies;
    ``show_reply`` None means a proper JSON object holding ``parameters``.
    """

    def do_GET(self) -> None:
        server: Any = self.server
        if self.path == "/api/version":
            self._send_raw(200, server.version_reply)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", 0)))
        server: Any = self.server
        if self.path != "/api/show":
            self._send(404, {"error": "not found"})
        elif not server.model_exists:
            self._send(404, {"error": "model 'codexprobe-qwen2.5-coder:1.5b' not found"})
        elif server.show_reply is not None:
            self._send_raw(200, server.show_reply)
        else:
            self._send(200, {"parameters": server.parameters})

    def _send(self, status: int, body: dict[str, Any]) -> None:
        self._send_raw(status, json.dumps(body).encode())

    def _send_raw(self, status: int, data: bytes) -> None:
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def fake_ollama() -> Iterator[Any]:
    server: Any = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
    server.version_reply, server.show_reply = b'{"version":"0.40.10"}', None
    server.model_exists, server.parameters = True, "num_ctx                        16384"
    server.url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def no_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the quickstart ever falls back to the mock server."""

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("qwen mode must never fall back to the mock")

    monkeypatch.setattr(quickstart, "MockResponsesServer", forbidden)


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_ollama_problem_names_each_missing_piece(fake_ollama: Any) -> None:
    assert quickstart.ollama_problem(fake_ollama.url) is None
    fake_ollama.parameters = "num_ctx 4096"
    assert "does not set num_ctx 16384" in str(quickstart.ollama_problem(fake_ollama.url))
    fake_ollama.model_exists = False
    assert "ollama create codexprobe-qwen2.5-coder:1.5b -f examples/ollama/Modelfile" in str(
        quickstart.ollama_problem(fake_ollama.url)
    )
    assert "not reachable" in str(quickstart.ollama_problem(f"http://127.0.0.1:{_closed_port()}/v1"))


def _qwen_argv(tmp_path: Path, url: str) -> list[str]:
    return [
        "--mode", "qwen", "--codex", sys.executable, "--codex-home", str(tmp_path / "home"),
        "--log-dir", str(tmp_path / "logs"), "--ollama-url", url,
    ]  # fmt: skip


def test_qwen_without_ollama_is_blocked(tmp_path: Path, capsys: pytest.CaptureFixture[str], no_mock: None) -> None:
    assert quickstart.main(_qwen_argv(tmp_path, f"http://127.0.0.1:{_closed_port()}/v1")) == quickstart.BLOCKED
    assert capsys.readouterr().out.startswith("BLOCKED: Ollama is not reachable at http://127.0.0.1:")
    assert not (tmp_path / "logs").exists()  # nothing ran


def test_qwen_without_the_model_is_blocked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], no_mock: None, fake_ollama: Any
) -> None:
    fake_ollama.model_exists = False
    assert quickstart.main(_qwen_argv(tmp_path, fake_ollama.url)) == quickstart.BLOCKED
    assert capsys.readouterr().out.startswith("BLOCKED: Ollama has no model codexprobe-qwen2.5-coder:1.5b")
    assert not (tmp_path / "logs").exists()


def test_qwen_with_ollama_ready_still_needs_the_right_codex(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], no_mock: None, fake_ollama: Any
) -> None:
    # Ollama and the model pass; Python, standing in for Codex, then fails the version check.
    assert quickstart.main(_qwen_argv(tmp_path, fake_ollama.url)) == quickstart.BLOCKED
    out = capsys.readouterr().out
    assert out.startswith("BLOCKED: ") and "needs 'codex-cli 0.164.0'" in out
    assert not (tmp_path / "logs").exists()


NOT_SHOW_OBJECT = 'is not a JSON object with a "parameters" string'
BAD_OLLAMA_REPLIES = [
    pytest.param("version_reply", b'{"version":"0.40.9"}', "reports version '0.40.9'; this quickstart needs '0.40.10'",
                 id="older version"),
    pytest.param("version_reply", b'{"version":"0.41.0"}', "reports version '0.41.0'", id="newer version"),
    pytest.param("version_reply", b'{"build":"f52b9e7"}', "reports version None", id="no version field"),
    pytest.param("version_reply", b'"0.40.10"', "did not answer /api/version with a JSON object", id="version not an object"),
    pytest.param("version_reply", b'{"version": "0.40', "did not answer /api/version with a JSON object", id="broken version JSON"),
    pytest.param("show_reply", b'{"parameters": "num_ctx 16384"', NOT_SHOW_OBJECT, id="broken show JSON"),
    pytest.param("show_reply", b'["num_ctx 16384"]', NOT_SHOW_OBJECT, id="show not an object"),
    pytest.param("show_reply", b'{"modelfile": "FROM qwen2.5-coder:1.5b"}', NOT_SHOW_OBJECT, id="no parameters"),
    pytest.param("show_reply", b'{"parameters": 16384}', NOT_SHOW_OBJECT, id="parameters not text"),
]  # fmt: skip


@pytest.mark.parametrize(("field", "body", "reason"), BAD_OLLAMA_REPLIES)
def test_ollama_problem_names_a_wrong_version_or_a_bad_reply(fake_ollama: Any, field: str, body: bytes, reason: str) -> None:
    assert quickstart.ollama_problem(fake_ollama.url) is None  # version 0.40.10 and a proper reply pass
    setattr(fake_ollama, field, body)
    assert reason in str(quickstart.ollama_problem(fake_ollama.url))


@pytest.mark.parametrize(("field", "body", "reason"), BAD_OLLAMA_REPLIES)
def test_a_wrong_version_or_a_bad_reply_blocks_the_quickstart_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    no_mock: None,
    fake_ollama: Any,
    field: str,
    body: bytes,
    reason: str,
) -> None:
    setattr(fake_ollama, field, body)
    assert quickstart.main(_qwen_argv(tmp_path, fake_ollama.url)) == quickstart.BLOCKED
    captured = capsys.readouterr()
    assert captured.out.startswith("BLOCKED: ") and reason in captured.out
    assert "Traceback" not in captured.out + captured.err
    assert not (tmp_path / "logs").exists()  # nothing ran


def _reply(seq: int, text: str | None, *, outcome: str = "completed", terminal: str | None = "response.completed",
           tools: tuple[str, ...] = ()) -> dict[str, Any]:  # fmt: skip
    items: list[dict[str, Any]] = []
    if text is not None:
        items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
    items += [{"type": "function_call", "name": name, "call_id": f"call-{seq}", "arguments": "{}"} for name in tools]
    return {
        "seq": seq,
        "outcome": outcome,
        "stream": {"reconstruction": {"terminal_event": terminal, "items": items}},
        "derived": {"tool_calls": [{"type": "function_call", "name": name, "call_id": f"call-{seq}"} for name in tools]},
    }


REPLY_CHECKS = [
    "codex exec finished in time with exit code 0",
    "LLM calls recorded",
    "every call ended with a complete response.completed",
    "the model's reply holds QWEN_LOCAL_OK",
    "no tool was called",
]


def test_a_complete_tool_free_reply_with_the_passphrase_passes() -> None:
    checks = quickstart.text_reply_checks([_reply(0, "QWEN_LOCAL_OK")], 0, False, "QWEN_LOCAL_OK")
    assert [check.name for check in checks] == REPLY_CHECKS
    assert all(check.passed for check in checks)


@pytest.mark.parametrize(
    ("calls", "exit_code", "timed_out", "failing"),
    [
        pytest.param([_reply(0, "Hello! How can I help?")], 0, False, [REPLY_CHECKS[3]], id="another reply"),
        pytest.param(
            [_reply(0, None, tools=("exec_command",))], 0, False, [REPLY_CHECKS[3], REPLY_CHECKS[4]], id="a tool call"
        ),
        pytest.param(
            [_reply(0, "QWEN_LOCAL_OK", outcome="stream_cut", terminal=None)], 0, False, [REPLY_CHECKS[2]], id="stream cut"
        ),
        pytest.param(
            [_reply(0, None, outcome="failed", terminal="response.failed")],
            0, False, [REPLY_CHECKS[2], REPLY_CHECKS[3]], id="model failed",
        ),  # fmt: skip
        pytest.param([], 0, False, REPLY_CHECKS[1:4], id="no calls"),
        pytest.param([_reply(0, "QWEN_LOCAL_OK")], 1, True, [REPLY_CHECKS[0]], id="timed out"),
    ],
)
def test_anything_less_fails_the_named_checks(
    calls: list[dict[str, Any]], exit_code: int, timed_out: bool, failing: list[str]
) -> None:
    checks = quickstart.text_reply_checks(calls, exit_code, timed_out, "QWEN_LOCAL_OK")
    assert [check.name for check in checks if not check.passed] == failing


# --- qwen coding task: the six facts ----------------------------------------------

CALC = "def add(a, b):\n    return a + b\n"
TEST = "import unittest\n\nfrom calc import add\n\n\nclass AddTest(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n"
FACTS = ["Backend Reached", "Response Captured", "Model Tool Call", "Arguments Valid", "Tool Executed", "Coding Task"]
BACKEND = "127.0.0.1:11434"
# The two tools that Codex 0.164.0 offers with the minimal tool set, in the form
# Codex sends them in a request.
CODEX_TOOLS = json.loads((PROJECT / "tests" / "data" / "codex-0.164.0-minimal-tools.json").read_text("utf-8"))


def _fc(call_id: str, arguments: Any, name: str = "exec_command") -> dict[str, Any]:
    return {"type": "function_call", "name": name, "call_id": call_id, "arguments": arguments}


def _code_call(
    seq: int,
    items: list[dict[str, Any]],
    outputs: tuple[str, ...] = (),
    *,
    peer: str = BACKEND,
    outcome: str = "completed",
    terminal: str | None = "response.completed",
) -> dict[str, Any]:
    """A recorded call as CodexProbe writes it, reduced to the fields the facts read."""
    return {
        "seq": seq,
        "outcome": outcome,
        "upstream": {"peer": peer},
        "request": {
            "body": {
                "json": {
                    "tools": copy.deepcopy(CODEX_TOOLS),
                    "input": [{"type": "function_call_output", "call_id": c, "output": "Process exited with code 0"} for c in outputs],
                }
            }
        },
        "stream": {"reconstruction": {"terminal_event": terminal, "items": items}},
        "derived": {"tool_calls": [{"type": "function_call", "name": i["name"], "call_id": i["call_id"]} for i in items
                                   if i["type"] == "function_call"]},
    }  # fmt: skip


def _good_calls() -> list[dict[str, Any]]:
    return [
        _code_call(0, [_fc("call-1", json.dumps({"cmd": "Set-Content -Path calc.py -Value 'def add(a, b): return a + b'"}))]),
        _code_call(1, [_fc("call-2", json.dumps({"cmd": "python -m unittest -v"}))], ("call-1",)),
        _code_call(2, [{"type": "message", "content": [{"type": "output_text", "text": "Done: 1 test passed."}]}], ("call-1", "call-2")),
    ]  # fmt: skip


def _event(item: dict[str, Any]) -> dict[str, Any]:
    return {"type": "item.completed", "item": item}


GOOD_EVENTS = [
    _event({"type": "file_change", "changes": [{"path": "calc.py", "kind": "add"}], "status": "completed"}),
    _event({"type": "command_execution", "command": "python -m unittest -v", "exit_code": 0, "status": "completed"}),
]


def _workspace(root: Path, kind: str) -> Path:
    root.mkdir()
    if kind != "empty":
        (root / "calc.py").write_text(CALC, "utf-8")
        (root / "test_calc.py").write_text(TEST.replace("5)", "6)") if kind == "failing" else TEST, "utf-8")
    return root


def _facts(calls: list[dict[str, Any]], events: list[dict[str, Any]], workspace: Path) -> list[Any]:
    return quickstart.coding_facts(calls, events, workspace, sys.executable, BACKEND)


def test_a_complete_coding_run_passes_all_six_facts(tmp_path: Path) -> None:
    facts = _facts(_good_calls(), GOOD_EVENTS, _workspace(tmp_path / "ws", "good"))
    assert [fact.name for fact in facts] == FACTS
    assert all(fact.passed for fact in facts), [f"{f.name}: {f.detail}" for f in facts if not f.passed]


def _say_done(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_code_call(0, [{"type": "message", "content": [{"type": "output_text", "text": "完成了"}]}])]


def _set_arguments(arguments: str, name: str = "exec_command") -> Any:
    def change(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        calls[0]["stream"]["reconstruction"]["items"] = [_fc("call-1", arguments, name)]
        return calls

    return change


def _set_schema(parameters: Any) -> Any:
    """Change the parameters schema that seq 0's request offers for exec_command (None removes it)."""

    def change(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        [tool] = [t for t in calls[0]["request"]["body"]["json"]["tools"] if t["name"] == "exec_command"]
        if parameters is None:
            del tool["parameters"]
        else:
            tool["parameters"] = parameters
        return calls

    return change


def _set_tool_name(name: Any) -> Any:
    """Give seq 0's request an exec_command tool definition whose name is ``name``."""

    def change(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        [tool] = [t for t in calls[0]["request"]["body"]["json"]["tools"] if t["name"] == "exec_command"]
        tool["name"] = name
        return calls

    return change


def _drop_output(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls[2]["request"]["body"]["json"]["input"] = calls[2]["request"]["body"]["json"]["input"][:1]  # call-2 unanswered
    return calls


def _elsewhere(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls[1]["upstream"]["peer"] = "203.0.113.7:11434"
    return calls


def _cut(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls[2]["outcome"], calls[2]["stream"]["reconstruction"]["terminal_event"] = "stream_cut", None
    return calls


FAILED_COMMAND = [
    _event({"type": "file_change", "changes": [{"path": "calc.py", "kind": "add"}], "status": "failed"}),
    _event({"type": "command_execution", "command": "python -m unittest -v", "exit_code": 1, "status": "failed"}),
]


@pytest.mark.parametrize(
    ("change", "events", "workspace", "failing"),
    [
        pytest.param(_say_done, [], "empty", ["Model Tool Call", "Arguments Valid", "Tool Executed", "Coding Task"],
                     id="says done without a tool call"),
        pytest.param(_set_arguments("{cmd: Set-Content calc.py"), GOOD_EVENTS, "good", ["Arguments Valid"], id="arguments not JSON"),
        pytest.param(_set_arguments('{"command": "dir"}'), GOOD_EVENTS, "good", ["Arguments Valid"], id="no cmd"),
        pytest.param(_set_arguments('{"cmd": "x"}', "apply_patch"), GOOD_EVENTS, "good", ["Arguments Valid"],
                     id="a tool that was not offered"),
        pytest.param(_set_arguments({"cmd": "dir"}), GOOD_EVENTS, "good", ["Arguments Valid"], id="arguments not a string"),
        pytest.param(_set_arguments('{"session_id": null, "chars": "dir\\n"}', "write_stdin"), GOOD_EVENTS, "good",
                     ["Arguments Valid"], id="write_stdin with a null session_id"),
        pytest.param(_set_arguments('{"cmd": "dir", "yield_time_ms": "30000"}'), GOOD_EVENTS, "good", ["Arguments Valid"],
                     id="an optional field of the wrong type"),
        pytest.param(_set_arguments('{"cmd": "dir", "max_output_tokens": true}'), GOOD_EVENTS, "good", ["Arguments Valid"],
                     id="a boolean where a number belongs"),
        pytest.param(_set_arguments('{"cmd": "dir", "chars": "x"}'), GOOD_EVENTS, "good", ["Arguments Valid"],
                     id="a field the schema does not allow"),
        pytest.param(_set_arguments('{"cmd": "dir", "sandbox_permissions": "full_access"}'), GOOD_EVENTS, "good",
                     ["Arguments Valid"], id="a value outside the enum"),
        pytest.param(_set_arguments('{"cmd": "git pull", "prefix_rule": ["git", 1]}'), GOOD_EVENTS, "good",
                     ["Arguments Valid"], id="an array item of the wrong type"),
        pytest.param(_set_arguments('{"cmd": "   "}'), GOOD_EVENTS, "good", ["Arguments Valid"], id="an empty cmd"),
        pytest.param(_set_schema(None), GOOD_EVENTS, "good", ["Arguments Valid"], id="the tool offered without a schema"),
        pytest.param(_set_schema("not a schema"), GOOD_EVENTS, "good", ["Arguments Valid"], id="a malformed schema"),
        pytest.param(_set_schema({"type": "object", "properties": {"cmd": {"type": "text"}}}), GOOD_EVENTS, "good",
                     ["Arguments Valid"], id="a schema with an unknown type"),
        pytest.param(_set_arguments('{"cmd": "dir", "yield_time_ms": NaN}'), GOOD_EVENTS, "good", ["Arguments Valid"],
                     id="NaN is not JSON"),
        pytest.param(_set_arguments('{"cmd": "dir", "yield_time_ms": Infinity}'), GOOD_EVENTS, "good", ["Arguments Valid"],
                     id="Infinity is not JSON"),
        pytest.param(_set_arguments('{"cmd": "dir", "yield_time_ms": -Infinity}'), GOOD_EVENTS, "good", ["Arguments Valid"],
                     id="-Infinity is not JSON"),
        pytest.param(_set_schema({"type": "object", "properties": {"cmd": {"type": "string"}}, "required": [[]]}), GOOD_EVENTS,
                     "good", ["Arguments Valid"], id="an unhashable required entry"),
        pytest.param(_set_schema({"type": "object", "properties": {"cmd": {"type": "string"}}, "required": [1]}), GOOD_EVENTS,
                     "good", ["Arguments Valid"], id="a required entry that is not a name"),
        pytest.param(_set_tool_name(["exec_command"]), GOOD_EVENTS, "good", ["Arguments Valid"],
                     id="a tool name that is not a string"),
        pytest.param(_drop_output, GOOD_EVENTS, "good", ["Tool Executed"], id="no matching function_call_output"),
        pytest.param(lambda calls: calls, FAILED_COMMAND, "good", ["Tool Executed"], id="the command failed"),
        pytest.param(lambda calls: calls, GOOD_EVENTS, "empty", ["Coding Task"], id="files missing"),
        pytest.param(lambda calls: calls, GOOD_EVENTS, "failing", ["Coding Task"], id="the tests fail when run again"),
        pytest.param(_elsewhere, GOOD_EVENTS, "good", ["Backend Reached"], id="a call went elsewhere"),
        pytest.param(_cut, GOOD_EVENTS, "good", ["Response Captured"], id="a stream was cut"),
    ],
)  # fmt: skip
def test_each_failure_fails_exactly_its_facts(
    tmp_path: Path, change: Any, events: list[dict[str, Any]], workspace: str, failing: list[str]
) -> None:
    facts = _facts(change(_good_calls()), events, _workspace(tmp_path / "ws", workspace))
    assert [fact.name for fact in facts if not fact.passed] == failing, [f"{f.name}: {f.detail}" for f in facts]


def test_a_valid_write_stdin_call_passes(tmp_path: Path) -> None:
    calls = _good_calls()
    calls[1]["stream"]["reconstruction"]["items"].append(
        _fc("call-3", '{"session_id": 3, "chars": "", "yield_time_ms": 1000}', "write_stdin")
    )
    calls[2]["request"]["body"]["json"]["input"].append({"type": "function_call_output", "call_id": "call-3", "output": ""})
    facts = _facts(calls, GOOD_EVENTS, _workspace(tmp_path / "ws", "good"))
    assert all(fact.passed for fact in facts), [f"{f.name}: {f.detail}" for f in facts if not f.passed]
    assert "3 call(s), each matching its tool's parameters schema" in facts[3].detail


def test_null_and_object_values_in_arguments_are_rejected(tmp_path: Path) -> None:
    # A null where a number belongs, and objects where a string and a number belong.
    null_value = _set_arguments('{"cmd": "dir", "yield_time_ms": null}')
    object_values = _set_arguments(json.dumps({"cmd": {"value": "dir"}, "max_output_tokens": {"value": 100}}))
    first = _facts(null_value(_good_calls()), GOOD_EVENTS, _workspace(tmp_path / "a", "good"))[3]
    second = _facts(object_values(_good_calls()), GOOD_EVENTS, _workspace(tmp_path / "b", "good"))[3]
    assert not first.passed and "arguments.yield_time_ms: expected number, got null" in first.detail
    assert not second.passed
    assert "arguments.cmd: expected string, got object" in second.detail
    assert "arguments.max_output_tokens: expected number, got object" in second.detail


def test_schema_problems_covers_the_rules_codex_schemas_use() -> None:
    [exec_command, write_stdin] = (tool["parameters"] for tool in CODEX_TOOLS)
    assert quickstart.schema_problems({"cmd": "dir", "login": False, "prefix_rule": ["git"], "yield_time_ms": 1.5}, exec_command) == []
    assert quickstart.schema_problems({"session_id": 7}, write_stdin) == []
    assert quickstart.schema_problems({}, write_stdin) == ["arguments: missing required 'session_id'"]
    assert quickstart.schema_problems({"session_id": True}, write_stdin) == [
        "arguments.session_id: expected number, got boolean"  # a JSON boolean is never a number
    ]
    assert quickstart.schema_problems({"cmd": "dir", "extra": 1}, exec_command) == ["arguments: unexpected 'extra'"]
    assert quickstart.schema_problems({"cmd": "dir", "sandbox_permissions": "x"}, exec_command) == [
        "arguments.sandbox_permissions: 'x' is not one of ['use_default', 'require_escalated']"
    ]
    assert quickstart.schema_problems({"cmd": "dir", "prefix_rule": "git"}, exec_command) == [
        "arguments.prefix_rule: expected array, got string"
    ]
    assert quickstart.schema_problems(["dir"], exec_command) == ["arguments: expected object, got array"]
    assert quickstart.schema_problems({"cmd": "dir"}, "nope") == ["arguments: the tool's schema here is not a JSON object"]
    assert quickstart.schema_problems(1, {"type": ["integer", "null"]}) == []
    assert quickstart.schema_problems(None, {"type": ["integer", "null"]}) == []
    assert quickstart.schema_problems(1.5, {"type": "integer"}) == ["arguments: expected integer, got number"]
    # A malformed schema is reported, never raised.
    assert quickstart.schema_problems({"cmd": "dir"}, {"type": "object", "required": [[]]}) == [
        "arguments: the schema's required is not a list of property names: [[]]"
    ]
    assert quickstart.schema_problems({"cmd": "dir"}, {"type": "object", "properties": []}) == [
        "arguments: the schema's properties are not a JSON object"
    ]
    assert quickstart.schema_problems({"cmd": "dir"}, {"type": [[]]}) == ["arguments: the schema names an unknown type [[]]"]


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_python_only_number_constants_are_not_json(tmp_path: Path, constant: str) -> None:
    calls = _set_arguments(f'{{"cmd": "dir", "max_output_tokens": {constant}}}')(_good_calls())
    facts = _facts(calls, GOOD_EVENTS, _workspace(tmp_path / "ws", "good"))
    assert [fact.name for fact in facts if not fact.passed] == ["Arguments Valid"]
    assert f"{constant} is not valid JSON" not in facts[3].detail  # the reason names the arguments, not Python's hook
    assert "the arguments are not valid JSON" in facts[3].detail


def test_the_report_lists_the_commands_and_what_codex_ran(tmp_path: Path) -> None:
    lines = quickstart.tool_call_lines(_set_arguments("{cmd: oops")(_good_calls()))
    assert lines[0] == "seq 0  exec_command (call-1): arguments '{cmd: oops'"
    assert lines[1] == "seq 1  exec_command (call-2): 'python -m unittest -v'"
    assert quickstart.execution_lines(GOOD_EVENTS) == [
        "file_change  completed  ['calc.py']",
        "command_execution  exit 0  completed  python -m unittest -v",
    ]
    stdout = tmp_path / "codex-stdout.jsonl"
    stdout.write_text("Reading additional input from stdin...\n" + "\n".join(json.dumps(e) for e in GOOD_EVENTS) + "\n[]\n", "utf-8")
    assert quickstart.codex_events(stdout) == GOOD_EVENTS  # text and non-object lines are skipped
    assert quickstart.codex_events(tmp_path / "missing.jsonl") == []


def test_backend_address_and_localhost() -> None:
    assert quickstart.backend_address("http://127.0.0.1:11434/v1") == "127.0.0.1:11434"
    assert quickstart.backend_address("https://ollama.example/v1") == "ollama.example:443"
    assert quickstart._reached("127.0.0.1:11434", quickstart.backend_address("http://localhost:11434/v1"))
    assert not quickstart._reached("127.0.0.1:11435", "localhost:11434")


# --- openai mode -----------------------------------------------------------------

OPENAI_CHECKS = [
    "Real requests to the ChatGPT Codex backend",
    "Every call ended with response.completed",
    "Server evidence in the response headers",
    "Coding Task",
]


def _openai_call(
    seq: int,
    *,
    url: str = f"{quickstart.OPENAI_UPSTREAM}/responses",
    source: str = "upstream",
    status: int = 200,
    headers: tuple[tuple[str, str], ...] = (("content-type", "text/event-stream"), ("x-request-id", "req_test")),
    outcome: str = "completed",
    terminal: str | None = "response.completed",
    text: str | None = None,
) -> dict[str, Any]:
    """A recorded call as CodexProbe writes it, reduced to the fields the openai checks read."""
    items = [] if text is None else [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]
    return {
        "seq": seq,
        "outcome": outcome,
        "upstream": {"url": url, "peer": "203.0.113.10:443"},
        "request": {"path": "/v1/responses", "body": {"json": {"model": "gpt-test"}}},
        "response": {"source": source, "status": status, "headers": [list(h) for h in headers]},
        "stream": {"reconstruction": {"terminal_event": terminal, "items": items}},
        "derived": {"tool_calls": []},
        "timing": {"first_byte_ms": 900.0, "total_ms": 3000.0},
    }


@pytest.fixture
def nothing_written(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the openai mode writes a config.toml, makes a temporary CODEX_HOME or starts the mock."""

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("openai mode must not write config.toml, create a CODEX_HOME or start the mock")

    monkeypatch.setattr(quickstart, "write_codex_config", forbidden)
    monkeypatch.setattr(quickstart, "MockResponsesServer", forbidden)
    monkeypatch.setattr(quickstart, "tempfile", SimpleNamespace(mkdtemp=forbidden))


def _openai_argv(tmp_path: Path, home: Path | None) -> list[str]:
    argv = ["--mode", "openai", "--codex", sys.executable, "--log-dir", str(tmp_path / "logs")]
    return argv + (["--codex-home", str(home)] if home is not None else [])


def test_openai_mode_parses() -> None:
    args = quickstart.parse_args(["--mode", "openai", "--codex-home", "signed-in"])
    assert args.mode == "openai" and args.codex_home == Path("signed-in")


def test_openai_needs_an_explicit_codex_home(tmp_path: Path, capsys: pytest.CaptureFixture[str], nothing_written: None) -> None:
    assert quickstart.main(_openai_argv(tmp_path, None)) == quickstart.BLOCKED
    assert capsys.readouterr().out.startswith("BLOCKED: openai mode needs --codex-home")
    assert not (tmp_path / "logs").exists()


def test_openai_refuses_the_default_codex_home(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, nothing_written: None
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".codex").mkdir()
    assert quickstart.main(_openai_argv(tmp_path, tmp_path / ".codex")) == quickstart.BLOCKED
    assert "is the default CODEX_HOME" in capsys.readouterr().out


def test_openai_needs_a_codex_home_that_exists(tmp_path: Path, capsys: pytest.CaptureFixture[str], nothing_written: None) -> None:
    assert quickstart.main(_openai_argv(tmp_path, tmp_path / "missing")) == quickstart.BLOCKED
    out = capsys.readouterr().out
    assert "does not exist; create it and sign in with ChatGPT" in out and " login" in out
    assert not (tmp_path / "missing").exists()  # nothing is created


def test_openai_refuses_a_codex_home_that_holds_a_config_toml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, nothing_written: None
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.toml"
    original = b'model = "codexprobe-mock"\n\x00\xff not even TOML'  # e.g. left by mock mode in a shared folder
    config.write_bytes(original)

    def must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("nothing may run when the CODEX_HOME holds a config.toml")

    for step in ("codex_version", "codex_login_status", "ProxyRecorder", "run_codex"):
        monkeypatch.setattr(quickstart, step, must_not_run)
    assert quickstart.main(_openai_argv(tmp_path, home)) == quickstart.BLOCKED
    out = capsys.readouterr().out
    assert out.startswith(f"BLOCKED: {home} holds a config.toml") and "no config.toml" in out
    assert config.read_bytes() == original  # left exactly as it was
    assert [path.name for path in home.iterdir()] == ["config.toml"]
    assert not (tmp_path / "logs").exists()  # no run directory was created


def test_openai_without_a_chatgpt_sign_in_is_blocked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, nothing_written: None
) -> None:
    home = tmp_path / "home"
    home.mkdir()

    def no_recorder(config: Any) -> None:
        raise AssertionError("nothing must start without a sign-in")

    monkeypatch.setattr(quickstart, "codex_version", lambda codex, env: quickstart.CODEX_VERSION)
    monkeypatch.setattr(quickstart, "codex_login_status", lambda codex, env: (False, "Not logged in"))
    monkeypatch.setattr(quickstart, "ProxyRecorder", no_recorder)
    assert quickstart.main(_openai_argv(tmp_path, home)) == quickstart.BLOCKED
    out = capsys.readouterr().out
    assert out.startswith(f"BLOCKED: {home}: Not logged in. Sign in with ChatGPT first:") and " login" in out
    assert not (tmp_path / "logs").exists() and list(home.iterdir()) == []


def test_login_status_is_read_without_repeating_credentials() -> None:
    assert quickstart.parse_login_status(0, "Logged in using ChatGPT\n") == (True, "Logged in using ChatGPT")
    signed_in, summary = quickstart.parse_login_status(0, "Logged in using an API key - sk-proj-***ABCD\n")
    assert not signed_in and "API key" in summary and "sk-" not in summary  # no API keys in this project
    assert quickstart.parse_login_status(1, "Not logged in\n") == (False, "Not logged in")
    assert quickstart.parse_login_status(1, "Error: CODEX_HOME points to x, but that path does not exist") == (
        False,
        "codex login status exited with 1",
    )
    # The real command, with Python standing in for Codex: it fails, so it is not a sign-in.
    assert quickstart.codex_login_status(sys.executable, dict(os.environ))[0] is False


def test_openai_overrides_point_codex_at_the_recorder_and_keep_the_sign_in() -> None:
    overrides = quickstart.openai_overrides("http://127.0.0.1:8135/v1")
    settings: dict[str, Any] = {}
    for override in overrides:
        key, _, value = override.partition("=")
        settings[key] = tomllib.loads(f"v = {value}")["v"]  # every value is valid TOML, as codex -c parses it
    provider = quickstart.OPENAI_PROVIDER
    assert provider not in ("openai", "ollama", "lmstudio")
    assert settings == {
        "model_provider": provider,
        f"model_providers.{provider}.name": "CodexProbe",
        f"model_providers.{provider}.base_url": "http://127.0.0.1:8135/v1",
        f"model_providers.{provider}.wire_api": "responses",
        f"model_providers.{provider}.requires_openai_auth": True,
        f"model_providers.{provider}.supports_websockets": False,
        f"model_providers.{provider}.request_max_retries": 0,
        f"model_providers.{provider}.stream_max_retries": 0,
        "web_search": "disabled",
        "features.unbounded_connection_retries": False,
        "features.plugins": False,
        "features.apps": False,
        "analytics.enabled": False,
        "windows.sandbox": "unelevated",
    }
    snippet = tomllib.loads(quickstart.config_snippet(overrides))  # the printed fragment says the same
    assert snippet["model_providers"][provider]["requires_openai_auth"] is True
    assert snippet["features"] == {"unbounded_connection_retries": False, "plugins": False, "apps": False}
    assert snippet["windows"] == {"sandbox": "unelevated"} and snippet["web_search"] == "disabled"


@pytest.mark.parametrize("task", ["code", "text"])
def test_the_openai_run_passes_its_settings_only_on_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, nothing_written: None, task: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    seen: dict[str, Any] = {}
    reply = quickstart.OPENAI_REPLY if task == "text" else None
    recorded = [_openai_call(0, text=reply)] if task == "text" else [_openai_call(0), _openai_call(1)]

    class FakeRecorder:
        def __init__(self, config: dict[str, Any]) -> None:
            seen["config"] = config
            self.run_dir = tmp_path / "logs" / "fake-run"

        def start(self) -> str:
            return "http://127.0.0.1:65000/v1"

        def stop(self) -> list[dict[str, Any]]:
            return copy.deepcopy(recorded)

    def fake_run_codex(command: list[str], env: dict[str, str], out: Path, timeout: float) -> tuple[int, bool, float]:
        seen["command"], seen["env"] = command, env
        if task == "code":
            workspace = Path(command[command.index("-C") + 1])
            (workspace / "calc.py").write_text(CALC, "utf-8")
            (workspace / "test_calc.py").write_text(TEST, "utf-8")
        return 0, False, 1.0

    def fake_load_session(path: Path) -> list[dict[str, Any]]:
        assert path == tmp_path / "logs" / "fake-run"  # the recorder's own run directory
        return copy.deepcopy(recorded)

    monkeypatch.setattr(quickstart, "ProxyRecorder", FakeRecorder)
    monkeypatch.setattr(quickstart, "run_codex", fake_run_codex)
    monkeypatch.setattr(quickstart, "load_session", fake_load_session)
    monkeypatch.setattr(quickstart, "codex_version", lambda codex, env: quickstart.CODEX_VERSION)
    monkeypatch.setattr(quickstart, "codex_login_status", lambda codex, env: (True, "Logged in using ChatGPT"))
    assert quickstart.main([*_openai_argv(tmp_path, home), "--task", task]) == quickstart.PASS
    assert seen["config"]["upstream"]["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert seen["config"]["upstream"]["auth"] == {"mode": "passthrough"}
    command = seen["command"]
    assert command[command.index("-s") + 1] == "workspace-write"
    assert command[-1] == (quickstart.OPENAI_PROMPT if task == "text" else quickstart.CODING_TASK_PROMPT)
    passed = [command[i + 1] for i, part in enumerate(command) if part == "-c"]
    assert passed == ["allow_login_shell=false", *quickstart.openai_overrides("http://127.0.0.1:65000/v1")]
    assert seen["env"]["CODEX_HOME"] == str(home) and "OPENAI_API_KEY" not in seen["env"]
    assert list(home.iterdir()) == []  # the signed-in CODEX_HOME is left exactly as it was
    out = capsys.readouterr().out
    assert "-c model_providers.codexprobe.requires_openai_auth=true" in out and "RESULT: PASS" in out
    assert "PASS  The log on disk reads back equal to stop()'s calls" in out


def test_a_genuine_openai_run_passes_every_check(tmp_path: Path) -> None:
    checks = quickstart.openai_checks([_openai_call(0), _openai_call(1)], _workspace(tmp_path / "ws", "good"), sys.executable)
    assert [check.name for check in checks] == OPENAI_CHECKS
    assert all(check.passed for check in checks), [f"{c.name}: {c.detail}" for c in checks if not c.passed]


@pytest.mark.parametrize(
    ("calls", "workspace", "failing"),
    [
        pytest.param([], "good", OPENAI_CHECKS[:3], id="exit code 0 but no calls"),
        pytest.param([_openai_call(0, outcome="stream_cut", terminal=None)], "good", [OPENAI_CHECKS[1]], id="a response cut short"),
        pytest.param([_openai_call(0, headers=(("content-type", "text/event-stream"),))], "good", [OPENAI_CHECKS[2]],
                     id="no server request id"),
        pytest.param([_openai_call(0, url="http://127.0.0.1:11434/v1/responses")], "good", [OPENAI_CHECKS[0]],
                     id="a call to another backend"),
        pytest.param([_openai_call(0, status=403, outcome="upstream_http_error", terminal=None)], "good", OPENAI_CHECKS[:2],
                     id="the backend refused the request"),
        pytest.param([_openai_call(0, source="proxy", status=502, headers=(("content-type", "application/json"),),
                                   outcome="proxy_error", terminal=None)], "good", OPENAI_CHECKS[:3], id="the proxy answered itself"),
        pytest.param([_openai_call(0)], "empty", [OPENAI_CHECKS[3]], id="files missing"),
        pytest.param([_openai_call(0)], "failing", [OPENAI_CHECKS[3]], id="the tests fail when run again"),
    ],
)  # fmt: skip
def test_each_openai_failure_fails_exactly_its_checks(
    tmp_path: Path, calls: list[dict[str, Any]], workspace: str, failing: list[str]
) -> None:
    checks = quickstart.openai_checks(calls, _workspace(tmp_path / "ws", workspace), sys.executable)
    assert [check.name for check in checks if not check.passed] == failing, [f"{c.name}: {c.detail}" for c in checks]


OPENAI_TEXT_CHECKS = [OPENAI_CHECKS[0], OPENAI_CHECKS[2], *[name.replace("QWEN_LOCAL_OK", "OPENAI_PROXY_OK") for name in REPLY_CHECKS]]


@pytest.mark.parametrize(
    ("calls", "failing"),
    [
        pytest.param([_openai_call(0, text="OPENAI_PROXY_OK")], [], id="a genuine reply"),
        pytest.param([_openai_call(0, text="Hello!")], [OPENAI_TEXT_CHECKS[5]], id="another reply"),
        pytest.param([_openai_call(0, text="OPENAI_PROXY_OK", headers=(("content-type", "text/event-stream"),))],
                     [OPENAI_TEXT_CHECKS[1]], id="no server request id"),
        pytest.param([_openai_call(0, text="OPENAI_PROXY_OK", url="http://127.0.0.1:11434/v1/responses")],
                     [OPENAI_TEXT_CHECKS[0]], id="a call to another backend"),
        pytest.param([], [*OPENAI_TEXT_CHECKS[:2], *OPENAI_TEXT_CHECKS[3:6]], id="no calls"),
    ],
)  # fmt: skip
def test_the_openai_text_reply_is_judged_like_the_qwen_one(calls: list[dict[str, Any]], failing: list[str]) -> None:
    checks = quickstart.openai_text_checks(calls, 0, False)
    assert [check.name for check in checks] == OPENAI_TEXT_CHECKS
    assert [check.name for check in checks if not check.passed] == failing, [f"{c.name}: {c.detail}" for c in checks]


def test_the_log_check_compares_the_log_on_disk_with_stop(tmp_path: Path) -> None:
    events = [ev_response_created("r"), ev_assistant_message("m", "logged"), ev_completed("r")]
    with MockResponsesServer([sse_response(events)]) as mock:
        recorder = ProxyRecorder({"upstream": {"base_url": mock.base_url, "auth": {"mode": "none"}}, "log_dir": str(tmp_path)})
        endpoint = recorder.start()
        try:
            with httpx.Client(trust_env=False, timeout=30) as client:
                reply = client.post(
                    f"{endpoint}/responses", headers={"content-type": "application/json"}, content=b'{"stream": true}'
                )
        finally:
            calls = recorder.stop()
    assert reply.status_code == 200
    assert [call["outcome"] for call in calls] == ["completed"] and quickstart.reply_text(calls[0]) == "logged"
    assert quickstart.log_check(calls, recorder.run_dir).passed  # one real record, read back as stop() returned it

    changed = copy.deepcopy(calls)
    changed[0]["outcome"] = "failed"  # the same number of calls and the same seq: only the content differs
    differ = quickstart.log_check(changed, recorder.run_dir)
    assert not differ.passed and differ.detail.endswith("they differ")

    [log] = [path for path in recorder.run_dir.iterdir() if path.suffix == ".jsonl"]
    log.write_bytes(log.read_bytes()[:-20])  # cut the record short, as a crash in mid-write would
    broken = quickstart.log_check(calls, recorder.run_dir)
    assert not broken.passed and log.name in broken.detail


# --- end to end, with the real Codex CLI (not in the default suite) -------------


@pytest.mark.e2e
def test_quickstart_mock_mode_with_the_real_codex(tmp_path: Path) -> None:
    codex = os.environ.get("CODEXPROBE_CODEX")
    if not codex:
        pytest.skip("set CODEXPROBE_CODEX to the Codex CLI 0.164.0 to run this test")
    argv = ["--mode", "mock", "--codex", codex, "--log-dir", os.environ.get("CODEXPROBE_LOG_DIR", str(tmp_path))]
    if os.environ.get("CODEXPROBE_CODEX_HOME"):
        argv += ["--codex-home", os.environ["CODEXPROBE_CODEX_HOME"]]
    assert quickstart.main(argv) == quickstart.PASS
