"""CodexProbe quickstart: record every LLM call of a real ``codex exec`` run.

Three modes, each running the real Codex CLI through CodexProbe:

* ``mock`` needs no credentials and no model. CodexProbe sits between Codex and
  a local mock of the OpenAI Responses API. The mock scripts a small coding
  task: Codex writes ``calc.py`` and a ``unittest`` test, runs the tests, and
  answers.
* ``qwen`` sends Codex to a local ``qwen2.5-coder:1.5b`` served by Ollama, built
  from ``examples/ollama/Modelfile`` as ``codexprobe-qwen2.5-coder:1.5b``. The
  model gets the same kind of coding task: write ``calc.py`` and a ``unittest``
  test, then run the tests. ``--task text`` instead asks it only to reply
  ``QWEN_LOCAL_OK`` and to use no tools: a smoke test.
* ``openai`` sends Codex, signed in with ChatGPT, to OpenAI's Codex backend
  (``https://chatgpt.com/backend-api/codex``) with the same coding task, or with
  ``--task text`` asks it only to reply ``OPENAI_PROXY_OK``. It needs
  ``--codex-home``: a separate CODEX_HOME where ``codex login`` was done and
  that holds no ``config.toml``. The quickstart only asks ``codex login status``. It never reads ``auth.json`` and writes
  nothing there: the provider settings go on the command line as ``-c``
  overrides. No API key is used.

It needs the Codex CLI 0.164.0 (``npm install -g @openai/codex@0.164.0``, or
pass ``--codex``) and this package (``pip install -e .``). From ``codex-probe``::

    python examples/quickstart.py --mode mock
    python examples/quickstart.py --mode qwen      # with `ollama serve` running
    python examples/quickstart.py --mode openai --codex-home <a signed-in CODEX_HOME>

What it does:

1. starts the backend, or checks that Ollama serves the model, then starts
   ``ProxyRecorder(config)`` in front of it;
2. prints the Codex ``config.toml`` and writes it into CODEX_HOME (a new
   temporary directory unless ``--codex-home`` names one). In openai mode it
   prints the settings and passes them as ``-c`` overrides instead;
3. runs the real ``codex exec -s workspace-write`` in a fresh workspace;
4. calls ``recorder.stop()`` and prints one line per recorded call;
5. checks the run independently, from the recorded calls and the workspace: the
   verdict never rests on Codex's exit code or final message alone. The mock and
   the text tasks also require exit code 0; the coding checks of qwen and openai
   do not use it.
   - mock: Codex exited with 0. The files exist, and their tests pass when run
     again here. Every tool call got its output back. Every call was recorded
     ``completed``, and every recorded body equals the mock's own bytes.
   - qwen, coding task: six facts, each reported on its own.
     - Backend Reached: every call reached the Ollama address.
     - Response Captured: every response ended with a complete
       ``response.completed``.
     - Model Tool Call: the model really called ``exec_command``.
     - Arguments Valid: the arguments satisfy the ``parameters`` schema that
       the same request offered for that tool.
     - Tool Executed: every call's output came back, and Codex really ran a
       command.
     - Coding Task: the files exist, and their tests pass when run again here.
   - qwen, text: Codex exited in time with exit code 0. At least one call was
     recorded, every one ending with a complete ``response.completed``. The
     model's reply holds ``QWEN_LOCAL_OK``, and no tool was called.
   - openai: every call went to the ChatGPT Codex backend and got HTTP 200
     from it. Every response ended with a complete ``response.completed``. Every
     response carries a request id set by the server. The files exist, and their
     tests pass when run again here. With ``--task text``, the backend checks
     plus the qwen text checks, for ``OPENAI_PROXY_OK``. Either way the log,
     read back from disk, must equal what ``stop()`` returned.

The mock is not a model: a mock PASS shows that Codex, CodexProbe and a
Responses server work together, not that a model can do the task.

Exit codes: 0 PASS; 1 FAIL; 2 bad arguments; 3 BLOCKED. BLOCKED means one of:
Codex is missing or not 0.164.0; the CODEX_HOME must not be used; Ollama is
missing or not 0.40.10; the model is missing or does not set num_ctx 16384; or,
in openai mode, there is no signed-in CODEX_HOME, or it holds a config.toml.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from codex_probe import ProxyRecorder, SessionLoadError, load_session
from codex_probe.codex_run import (
    CODEX_VERSION,
    TIMEOUT_S,
    CodexHomeError,
    base_python,
    check_codex_home,
    codex_config,
    codex_env,
    codex_version,
    exec_command_line,
    run_codex,
    write_codex_config,
)
from codex_probe.e2e_validation import Check, Report, RunEvidence, run_workspace_tests, verify
from codex_probe.mock_server import CODING_TASK_FILES, CODING_TASK_PROMPT, MockResponsesServer, coding_task

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = PROJECT_DIR / "runs"  # ignored by .gitignore
PASS, FAIL, BLOCKED = 0, 1, 3

OLLAMA_VERSION = "0.40.10"  # the Ollama release the qwen mode requires
QWEN_URL = "http://127.0.0.1:11434/v1"  # Ollama's OpenAI-compatible API
QWEN_MODEL = "codexprobe-qwen2.5-coder:1.5b"  # built from examples/ollama/Modelfile
QWEN_CONTEXT = 16384  # the Modelfile's num_ctx; Codex's model_context_window must match
QWEN_IDLE_TIMEOUT_MS = 600_000  # Ollama sends no byte until the first token
QWEN_TIMEOUT_S = 3600  # wall-clock limit for one codex exec against a local model
QWEN_REPLY = "QWEN_LOCAL_OK"
QWEN_PROMPT = f"Reply with {QWEN_REPLY}. Do not use tools."
# The same kind of task as the mock's, written as a user would ask for it. The
# files it names are CODING_TASK_FILES. The prompt also names the tool to use and
# how to write each file.
QWEN_CODE_PROMPT = (
    "Use the exec_command tool for each step. Create calc.py with a function add(a, b) that returns a + b, "
    "and test_calc.py with a unittest test for add; write each file with a PowerShell Set-Content command. "
    "Then run the tests with python -m unittest -v."
)

OPENAI_UPSTREAM = "https://chatgpt.com/backend-api/codex"  # Codex's backend for a ChatGPT sign-in
# The provider id must not be a built-in one (openai, ollama, lmstudio), and its name
# must not be "OpenAI", which turns on OpenAI-only paths such as WebSockets and zstd.
OPENAI_PROVIDER = "codexprobe"
OPENAI_REPLY = "OPENAI_PROXY_OK"
OPENAI_PROMPT = f"Reply with {OPENAI_REPLY}. Do not use tools."
# Response headers that only the real server sets, not CodexProbe or the mock.
SERVER_HEADERS = frozenset({"x-request-id", "x-oai-request-id", "openai-processing-ms"})


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command line. Bad arguments exit with 2."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mode",
        required=True,
        choices=["mock", "qwen", "openai"],
        help="mock: no credentials, no model; qwen: a local Qwen via Ollama; openai: OpenAI's Codex backend via a ChatGPT sign-in",
    )
    parser.add_argument("--codex", help="the Codex CLI to run (default: codex on PATH)")
    parser.add_argument(
        "--codex-home",
        type=Path,
        help="CODEX_HOME for the run. mock and qwen write config.toml there (default: a new temporary directory); "
        "openai needs a separate one where `codex login` was done and no config.toml exists, and writes nothing there",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="where the logs, the mock capture and the workspace go (default: %(default)s)"
    )
    parser.add_argument(
        "--timeout",
        type=_seconds,
        help=f"seconds before codex exec is killed (default: {TIMEOUT_S} for mock and openai, {QWEN_TIMEOUT_S} for qwen)",
    )
    parser.add_argument(
        "--ollama-url",
        type=_ollama_url,
        default=QWEN_URL,
        help="qwen mode: the base URL of Ollama's OpenAI-compatible API, ending in /v1 (default: %(default)s)",
    )
    parser.add_argument(
        "--task",
        "--qwen-task",  # its first name, from when only the qwen mode had a text task
        dest="task",
        choices=["code", "text"],
        default="code",
        help="qwen and openai modes: the coding task (default), or a text reply only, as a smoke test",
    )
    args = parser.parse_args(argv)
    if args.mode == "mock" and args.task != "code":
        parser.error("--task text is for the qwen and openai modes; the mock always runs the coding task")
    return args


def _seconds(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"{text} is not a positive number of seconds")
    return value


def _ollama_url(text: str) -> str:
    parts = urlsplit(text)
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.query
        or parts.fragment
        or parts.path.rstrip("/") != "/v1"
    ):
        raise argparse.ArgumentTypeError(f"{text} is not an http(s) URL of the form http://host:port/v1")
    return text.rstrip("/")


def ollama_problem(base_url: str, model: str = QWEN_MODEL) -> str | None:
    """Why Ollama at ``base_url`` cannot serve ``model`` for this run, or None if it can.

    It must be Ollama :data:`OLLAMA_VERSION`, and the model must set num_ctx
    :data:`QWEN_CONTEXT`. Nothing is pulled, created or upgraded: a missing model
    is reported, with the command that builds it.
    """
    root = base_url.removesuffix("/v1")
    try:
        with httpx.Client(trust_env=False, timeout=10) as client:
            version = client.get(f"{root}/api/version")
            if version.status_code != 200:
                return f"Ollama at {root} answered HTTP {version.status_code} to /api/version"
            reported = _json_object(version)
            if reported is None:
                return f"Ollama at {root} did not answer /api/version with a JSON object"
            if reported.get("version") != OLLAMA_VERSION:
                return f"Ollama at {root} reports version {reported.get('version')!r}; this quickstart needs {OLLAMA_VERSION!r}"
            show = client.post(f"{root}/api/show", json={"model": model})
    except httpx.HTTPError as exc:
        return f"Ollama is not reachable at {root} ({type(exc).__name__}); start it with `ollama serve`"
    if show.status_code == 404:
        return f"Ollama has no model {model}; build it with `ollama create {model} -f examples/ollama/Modelfile`"
    if show.status_code != 200:
        return f"Ollama answered HTTP {show.status_code} when asked about {model}"
    details = _json_object(show)
    parameters = details.get("parameters") if details is not None else None
    if not isinstance(parameters, str):
        return f"Ollama's /api/show reply for {model} is not a JSON object with a \"parameters\" string"
    if not re.search(rf"\bnum_ctx\s+{QWEN_CONTEXT}\b", parameters):
        return f"{model} does not set num_ctx {QWEN_CONTEXT}; rebuild it from examples/ollama/Modelfile"
    return None


def _json_object(response: httpx.Response) -> dict[str, Any] | None:
    """The response body as a JSON object, or None if it is anything else."""
    try:
        data = response.json()
    except ValueError:  # invalid JSON, or bytes that are not text
        return None
    return data if isinstance(data, dict) else None


def qwen_codex_config(endpoint: str) -> str:
    """config.toml for the qwen mode: the local model's context, a long idle timeout, the minimal tool set."""
    return codex_config(
        endpoint,
        model=QWEN_MODEL,
        context_window=QWEN_CONTEXT,
        stream_idle_timeout_ms=QWEN_IDLE_TIMEOUT_MS,
        minimal_tools=True,
    )


def summarize(calls: list[dict[str, Any]]) -> list[str]:
    """One line per recorded LLM call: seq, the model requested, outcome, tool calls and latency."""
    lines = []
    for call in calls:
        request = call["request"]["body"].get("json")
        model = request.get("model") if isinstance(request, dict) else None
        tools = ", ".join(f"{tool['name']} ({tool['call_id']})" for tool in call["derived"]["tool_calls"]) or "none"
        timing = call["timing"]
        lines.append(
            f"seq {call['seq']}  model {model}  {call['outcome']}  tool calls: {tools}  "
            f"first byte {_ms(timing.get('first_byte_ms'))}, total {_ms(timing.get('total_ms'))}"
        )
    return lines


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


def reply_text(call: dict[str, Any]) -> str:
    """The text of the assistant messages in one recorded response, as reassembled by CodexProbe."""
    items = ((call.get("stream") or {}).get("reconstruction") or {}).get("items") or []
    return "\n".join(
        part.get("text", "")
        for item in items
        if item.get("type") == "message"
        for part in item.get("content", [])
        if isinstance(part, dict) and part.get("type") == "output_text"
    )


def text_reply_checks(calls: list[dict[str, Any]], exit_code: int, timed_out: bool, reply: str) -> list[Check]:
    """A text smoke test (qwen or openai), judged from the recorded calls, not from Codex's own output."""
    finished = [c for c in calls if c["outcome"] == "completed" and _terminal(c) == "response.completed"]
    texts = [reply_text(c) for c in calls]
    tool_calls = [tool["name"] for c in calls for tool in c["derived"]["tool_calls"]]
    return [
        Check(
            "codex exec finished in time with exit code 0",
            exit_code == 0 and not timed_out,
            "killed at the time limit" if timed_out else f"exit code {exit_code}",
        ),
        Check("LLM calls recorded", bool(calls), f"{len(calls)} call(s)"),
        Check(
            "every call ended with a complete response.completed",
            bool(calls) and len(finished) == len(calls),
            ", ".join(f"seq {c['seq']}: {c['outcome']} ({_terminal(c)})" for c in calls) or "no calls",
        ),
        Check(f"the model's reply holds {reply}", any(reply in text for text in texts), f"replies: {texts!r}"[:300]),
        Check("no tool was called", not tool_calls, f"tool calls: {tool_calls}" if tool_calls else "none"),
    ]


def _terminal(call: dict[str, Any]) -> str | None:
    return ((call.get("stream") or {}).get("reconstruction") or {}).get("terminal_event")


def backend_address(base_url: str) -> str:
    """The ``host:port`` that CodexProbe should record as the upstream peer for ``base_url``."""
    parts = urlsplit(base_url)
    return f"{parts.hostname}:{parts.port or (443 if parts.scheme == 'https' else 80)}"


def codex_events(path: Path) -> list[dict[str, Any]]:
    """Codex's ``--json`` events, one per line. Lines that are not JSON objects are skipped."""
    events: list[dict[str, Any]] = []
    lines = path.read_text("utf-8", errors="replace").splitlines() if path.is_file() else []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def tool_calls(calls: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every function call the model made, with the recorded call whose response holds it."""
    return [
        (call, item)
        for call in calls
        for item in ((call.get("stream") or {}).get("reconstruction") or {}).get("items") or []
        if item.get("type") == "function_call"
    ]


def tool_call_lines(calls: list[dict[str, Any]]) -> list[str]:
    """One line per function call: what the model asked Codex to run."""
    lines = []
    for call, item in tool_calls(calls):
        arguments = item.get("arguments")
        try:
            cmd = json.loads(arguments).get("cmd")
        except (TypeError, ValueError, AttributeError):
            cmd = None
        shown = repr(cmd) if isinstance(cmd, str) else f"arguments {arguments!r}"
        lines.append(f"seq {call['seq']}  {item.get('name')} ({item.get('call_id')}): {shown}"[:400])
    return lines


def _executions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The commands and file changes Codex itself reports as finished."""
    return [
        event["item"]
        for event in events
        if event.get("type") == "item.completed"
        and isinstance(event.get("item"), dict)
        and event["item"].get("type") in ("command_execution", "file_change")
    ]


def execution_lines(events: list[dict[str, Any]]) -> list[str]:
    """One line per command or file change that Codex reports."""
    lines = []
    for item in _executions(events):
        if item["type"] == "command_execution":
            lines.append(f"command_execution  exit {item.get('exit_code')}  {item.get('status')}  {item.get('command')}"[:400])
        else:
            lines.append(f"file_change  {item.get('status')}  {[c.get('path') for c in item.get('changes', [])]}"[:400])
    return lines


_JSON_TYPES = {
    "string": lambda value: isinstance(value, str),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "array": lambda value: isinstance(value, list),
    "object": lambda value: isinstance(value, dict),
    "null": lambda value: value is None,
}


def _json_type(value: Any) -> str:
    """The JSON type name of a decoded value; a bool is a boolean, never a number."""
    for name in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _JSON_TYPES[name](value):
            return "number" if name == "integer" else name
    return type(value).__name__


def schema_problems(value: Any, schema: Any, path: str = "arguments") -> list[str]:
    """Check ``value`` against the part of JSON Schema that Codex's tool schemas use.

    That is ``type``, ``properties``, ``required``, ``additionalProperties: false``,
    array ``items`` and ``enum``. A schema that is not an object, or names an unknown
    type, is reported as a problem rather than raising.
    """
    if not isinstance(schema, dict):
        return [f"{path}: the tool's schema here is not a JSON object"]
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not all(isinstance(t, str) and t in _JSON_TYPES for t in types):
            return [f"{path}: the schema names an unknown type {expected!r}"]
        if not any(_JSON_TYPES[t](value) for t in types):
            return [f"{path}: expected {' or '.join(types)}, got {_json_type(value)}"]
    problems = []
    if "enum" in schema:
        allowed = schema["enum"]
        if not isinstance(allowed, list):
            return [f"{path}: the schema's enum is not a list"]
        if value not in allowed:
            problems.append(f"{path}: {value!r} is not one of {allowed}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict):
            return [f"{path}: the schema's properties are not a JSON object"]
        if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
            return [f"{path}: the schema's required is not a list of property names: {required!r}"[:300]]
        problems += [f"{path}: missing required {name!r}" for name in required if name not in value]
        if schema.get("additionalProperties") is False:
            problems += [f"{path}: unexpected {name!r}" for name in value if name not in properties]
        for name, item in value.items():
            if name in properties:
                problems += schema_problems(item, properties[name], f"{path}.{name}")
    if isinstance(value, list) and "items" in schema:
        for index, element in enumerate(value):
            problems += schema_problems(element, schema["items"], f"{path}[{index}]")
    return problems


def _reject_constant(name: str) -> Any:
    """``json.loads`` hook: NaN, Infinity and -Infinity are not JSON, though Python accepts them."""
    raise ValueError(f"{name} is not valid JSON")


def _argument_problems(call: dict[str, Any], item: dict[str, Any]) -> list[str]:
    """Problems with one function call's arguments, judged by the ``parameters`` schema
    that the same request offered for that tool."""
    where = f"seq {call['seq']} {item.get('call_id')}"
    request = call["request"]["body"].get("json")
    tools = request.get("tools") if isinstance(request, dict) else None
    offered = (
        {tool["name"]: tool for tool in tools if isinstance(tool, dict) and isinstance(tool.get("name"), str)}
        if isinstance(tools, list)
        else {}
    )
    name = item.get("name")
    if not isinstance(name, str) or name not in offered:
        return [f"{where}: {name!r} is not one of the offered tools {sorted(offered)}"]
    raw = item.get("arguments")
    if not isinstance(raw, str):
        return [f"{where}: the arguments are a JSON {_json_type(raw)}, not a string holding a JSON object"]
    try:
        arguments = json.loads(raw, parse_constant=_reject_constant)
    except ValueError:
        return [f"{where}: the arguments are not valid JSON: {raw!r}"[:300]]
    if not isinstance(arguments, dict):
        return [f"{where}: the arguments are not a JSON object"]
    schema = offered[name].get("parameters")
    if not isinstance(schema, dict):
        return [f"{where}: the request offered {name!r} without a parameters schema to check the arguments against"]
    problems = [f"{where}: {problem}" for problem in schema_problems(arguments, schema)]
    cmd = arguments.get("cmd")
    if name == "exec_command" and isinstance(cmd, str) and not cmd.strip():
        problems.append(f"{where}: exec_command with an empty cmd")
    return problems


def _outputs_after(calls: list[dict[str, Any]], seq: int) -> set[Any]:
    """The call_ids answered by a function_call_output in any request after ``seq``."""
    answered = set()
    for call in calls:
        if call["seq"] > seq:
            request = call["request"]["body"].get("json") or {}
            for entry in request.get("input", []):
                if isinstance(entry, dict) and entry.get("type") == "function_call_output":
                    answered.add(entry.get("call_id"))
    return answered


def _reached(peer: Any, backend: str) -> bool:
    host, _, port = backend.rpartition(":")
    return peer == backend or (host == "localhost" and peer in (f"127.0.0.1:{port}", f"::1:{port}"))


def coding_facts(
    calls: list[dict[str, Any]], events: list[dict[str, Any]], workspace: Path, python: str, backend: str
) -> list[Check]:
    """The qwen coding task, judged one fact at a time.

    The evidence is the calls CodexProbe recorded, Codex's own JSON events, and the
    workspace, whose tests are run again here with ``python``. Codex's exit code and
    final message are not used.
    """
    issued = tool_calls(calls)
    finished = [c for c in calls if c["outcome"] == "completed" and _terminal(c) == "response.completed"]
    argument_problems = [problem for call, item in issued for problem in _argument_problems(call, item)]
    unanswered = [str(item.get("call_id")) for call, item in issued if item.get("call_id") not in _outputs_after(calls, call["seq"])]
    executions = _executions(events)
    succeeded = [
        e for e in executions if e.get("status") == "completed" and (e["type"] == "file_change" or e.get("exit_code") == 0)
    ]
    failed = [f"{e['type']} {e.get('status')} exit {e.get('exit_code')}" for e in executions if e not in succeeded]
    peers = sorted({str(c["upstream"]["peer"]) for c in calls})
    return [
        Check(
            "Backend Reached",
            bool(calls) and all(_reached(c["upstream"]["peer"], backend) for c in calls),
            f"{len(calls)} call(s); upstream peer(s) {peers}, expected {backend}",
        ),
        Check(
            "Response Captured",
            bool(calls) and len(finished) == len(calls),
            ", ".join(f"seq {c['seq']}: {c['outcome']} ({_terminal(c)})" for c in calls) or "no calls",
        ),
        Check(
            "Model Tool Call",
            any(item.get("name") == "exec_command" for _, item in issued),
            f"{len(issued)} function call(s): {[item.get('name') for _, item in issued]}",
        ),
        Check(
            "Arguments Valid",
            bool(issued) and not argument_problems,
            "; ".join(argument_problems)
            or (f"{len(issued)} call(s), each matching its tool's parameters schema" if issued else "no tool calls to check"),
        ),
        Check(
            "Tool Executed",
            bool(issued) and not unanswered and bool(succeeded),
            "no tool calls to execute"
            if not issued
            else (f"no function_call_output for {', '.join(unanswered)}; " if unanswered else "")
            + f"Codex ran {len(succeeded)} successfully"
            + (f", {len(failed)} failed or declined ({'; '.join(failed)})" if failed else ""),
        ),
        coding_task_check(workspace, python),
    ]


def coding_task_check(workspace: Path, python: str) -> Check:
    """The coding task's own result: both files exist, and their tests pass when run again here."""
    missing = [name for name in CODING_TASK_FILES if not (workspace / name).is_file()]
    if missing:
        return Check("Coding Task", False, f"missing from the workspace: {', '.join(missing)}")
    tests_passed, detail = run_workspace_tests(workspace, python)
    sizes = ", ".join(f"{name} ({(workspace / name).stat().st_size} bytes)" for name in CODING_TASK_FILES)
    return Check("Coding Task", tests_passed, f"{sizes}; {detail}")


# --- openai mode -------------------------------------------------------------


def login_command(codex: str, codex_home: Path) -> str:
    """The commands that sign Codex in with ChatGPT inside ``codex_home``. The user runs them."""
    if os.name == "nt":
        return f'New-Item -ItemType Directory -Force "{codex_home}"; $env:CODEX_HOME = "{codex_home}"; & "{codex}" login'
    return f'mkdir -p "{codex_home}" && CODEX_HOME="{codex_home}" "{codex}" login'


def openai_home_problem(codex_home: Path | None, codex: str) -> str | None:
    """Why ``codex_home`` cannot be used for the openai mode, or None.

    Nothing is created, and nothing in ``codex_home`` is opened: a ``config.toml``
    there is refused by its name alone. Codex would apply it to the run alongside
    the ``-c`` overrides, and they cannot vouch for what else it sets.
    """
    if codex_home is None:
        return "openai mode needs --codex-home: a CODEX_HOME where `codex login` was done with ChatGPT"
    if codex_home.resolve() == (Path.home() / ".codex").resolve():
        return f"{codex_home} is the default CODEX_HOME; sign in into a separate one: {login_command(codex, codex_home)}"
    if not codex_home.is_dir():
        return f"{codex_home} does not exist; create it and sign in with ChatGPT: {login_command(codex, codex_home)}"
    if (codex_home / "config.toml").exists(follow_symlinks=False):  # any entry of that name, even a broken link
        return (
            f"{codex_home} holds a config.toml, which Codex would apply to this run. The openai quickstart needs a "
            "separate CODEX_HOME, signed in with ChatGPT, that has no config.toml (the file was not opened or changed)"
        )
    return None


def parse_login_status(returncode: int, output: str) -> tuple[bool, str]:
    """Read ``codex login status``: (signed in with ChatGPT, a summary safe to print).

    The summary never repeats the raw output, which for an API-key login shows part of the key.
    """
    if returncode == 0 and "Logged in using ChatGPT" in output:
        return True, "Logged in using ChatGPT"
    if "API key" in output:
        return False, "signed in with an API key, which this project does not use"
    if "Not logged in" in output:
        return False, "Not logged in"
    return False, f"codex login status exited with {returncode}"


def codex_login_status(codex: str, env: dict[str, str]) -> tuple[bool, str]:
    """Run ``codex login status`` in ``env``'s CODEX_HOME, with stdin closed and a time limit."""
    try:
        done = subprocess.run(
            [codex, "login", "status"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            env=env,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"codex login status could not run ({type(exc).__name__})"
    return parse_login_status(done.returncode, done.stdout + done.stderr)


def openai_overrides(endpoint: str) -> list[str]:
    """The provider and safety settings, as ``codex exec -c key=value`` overrides (TOML values).

    They point Codex at the recorder, keep the ChatGPT sign-in (``requires_openai_auth``),
    use HTTP and SSE rather than WebSockets, and turn off retries, plugins, apps,
    analytics and web search. Nothing is written to the signed-in CODEX_HOME.
    """
    provider = f"model_providers.{OPENAI_PROVIDER}"
    return [
        f'model_provider="{OPENAI_PROVIDER}"',
        f'{provider}.name="CodexProbe"',
        f'{provider}.base_url="{endpoint}"',
        f'{provider}.wire_api="responses"',
        f"{provider}.requires_openai_auth=true",
        f"{provider}.supports_websockets=false",
        f"{provider}.request_max_retries=0",
        f"{provider}.stream_max_retries=0",
        'web_search="disabled"',
        "features.unbounded_connection_retries=false",
        "features.plugins=false",
        "features.apps=false",
        "analytics.enabled=false",
        'windows.sandbox="unelevated"',
    ]


def config_snippet(overrides: list[str]) -> str:
    """The same settings as a ``config.toml`` fragment, to read; nothing is written."""
    root: dict[str, Any] = {}
    for override in overrides:
        key, _, value = override.partition("=")
        *tables, name = key.split(".")
        table = root
        for part in tables:
            table = table.setdefault(part, {})
        table[name] = value  # the TOML value, as written in the override
    lines: list[str] = []

    def emit(prefix: str, table: dict[str, Any]) -> None:
        values = [(k, v) for k, v in table.items() if not isinstance(v, dict)]
        if prefix and values:
            lines.append(f"\n[{prefix}]")
        lines.extend(f"{k} = {v}" for k, v in values)
        for k, v in table.items():
            if isinstance(v, dict):
                emit(f"{prefix}.{k}" if prefix else k, v)

    emit("", root)
    return "\n".join(lines).strip() + "\n"


def _server_headers(call: dict[str, Any]) -> list[str]:
    return sorted({str(name).lower() for name, _ in call["response"]["headers"] or []} & SERVER_HEADERS)


def openai_backend_checks(calls: list[dict[str, Any]]) -> tuple[Check, Check]:
    """Every call got HTTP 200 from the ChatGPT Codex backend, with a request id that only the real server sets."""
    to_backend = [
        c
        for c in calls
        if c["response"]["source"] == "upstream"
        and c["response"]["status"] == 200
        and str(c["upstream"]["url"]) == f"{OPENAI_UPSTREAM}/responses"
    ]
    with_ids = [c for c in calls if _server_headers(c)]
    reached = Check(
        "Real requests to the ChatGPT Codex backend",
        bool(calls) and len(to_backend) == len(calls),
        f"{len(to_backend)} of {len(calls)} call(s) went to {OPENAI_UPSTREAM}/responses and got HTTP 200 from it; "
        f"statuses {[(c['response']['source'], c['response']['status']) for c in calls]}",
    )
    evidence = Check(
        "Server evidence in the response headers",
        bool(calls) and len(with_ids) == len(calls),
        ", ".join(f"seq {c['seq']}: {_server_headers(c) or 'none'}" for c in calls) or "no calls",
    )
    return reached, evidence


def openai_checks(calls: list[dict[str, Any]], workspace: Path, python: str) -> list[Check]:
    """The openai coding run, judged from the recorded calls and the workspace, not from Codex's exit code."""
    reached, evidence = openai_backend_checks(calls)
    finished = [c for c in calls if c["outcome"] == "completed" and _terminal(c) == "response.completed"]
    return [
        reached,
        Check(
            "Every call ended with response.completed",
            bool(calls) and len(finished) == len(calls),
            ", ".join(f"seq {c['seq']}: {c['outcome']} ({_terminal(c)})" for c in calls) or "no calls",
        ),
        evidence,
        coding_task_check(workspace, python),
    ]


def openai_text_checks(calls: list[dict[str, Any]], exit_code: int, timed_out: bool) -> list[Check]:
    """The openai text smoke test: the backend checks, then the text checks for :data:`OPENAI_REPLY`."""
    return [*openai_backend_checks(calls), *text_reply_checks(calls, exit_code, timed_out, OPENAI_REPLY)]


def log_check(calls: list[dict[str, Any]], run_dir: Path) -> Check:
    """The log on disk equals what ``stop()`` returned.

    ``load_session`` also checks every record against the schema, which requires
    every credential header to be redacted.
    """
    name = "The log on disk reads back equal to stop()'s calls"
    try:
        logged = load_session(run_dir)
    except SessionLoadError as exc:
        return Check(name, False, str(exc))
    same = logged == calls
    return Check(name, same, f"{len(logged)} call(s) read back, {len(calls)} returned by stop()" + ("" if same else "; they differ"))


def blocked(reason: str) -> int:
    print(f"BLOCKED: {reason}")
    return BLOCKED


def main(argv: list[str] | None = None) -> int:
    """Run the quickstart and return its exit code."""
    args = parse_args(argv)
    timeout = args.timeout or (QWEN_TIMEOUT_S if args.mode == "qwen" else TIMEOUT_S)

    # What the run needs. Without it the result is BLOCKED, never a substitute.
    codex = shutil.which(args.codex or "codex")
    if codex is None:
        return blocked(f"the Codex CLI was not found ({args.codex or 'codex on PATH'}); install @openai/codex@0.164.0")
    if args.mode == "openai":
        problem = openai_home_problem(args.codex_home, codex)
        if problem is not None:
            return blocked(problem)
        codex_home: Path = args.codex_home  # the signed-in CODEX_HOME, used as it is: nothing is written there
    else:
        codex_home = args.codex_home or Path(tempfile.mkdtemp(prefix="codexprobe-codex-home-"))
        try:
            check_codex_home(codex_home)
        except CodexHomeError as exc:
            return blocked(str(exc))
        if args.mode == "qwen":
            problem = ollama_problem(args.ollama_url)
            if problem is not None:
                return blocked(problem)
        codex_home.mkdir(parents=True, exist_ok=True)
    env = codex_env(codex_home)
    version = codex_version(codex, env)
    if version != CODEX_VERSION:
        return blocked(f"{codex} reports {version!r}; this quickstart needs {CODEX_VERSION!r}")
    if args.mode == "openai":
        signed_in, status = codex_login_status(codex, env)
        if not signed_in:
            return blocked(f"{codex_home}: {status}. Sign in with ChatGPT first: {login_command(codex, codex_home)}")

    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    run_root = args.log_dir / (f"quickstart-{stamp}" if args.mode == "mock" else f"quickstart-{args.mode}-{stamp}")
    workspace = run_root / "workspace"
    workspace.mkdir(parents=True)  # new and empty: anything in it afterwards came from Codex
    if args.mode == "mock":
        return run_mock(codex, codex_home, env, timeout, run_root, workspace)
    if args.mode == "openai":
        return run_openai(codex, env, timeout, run_root, workspace, args.task)
    return run_qwen(codex, codex_home, env, timeout, run_root, workspace, args.ollama_url, args.task)


def run_mock(codex: str, codex_home: Path, env: dict[str, str], timeout: float, run_root: Path, workspace: Path) -> int:
    """Codex -> CodexProbe -> the mock's coding task, then the Phase 2 checks."""
    capture_dir = run_root / "mock-capture"
    with MockResponsesServer(coding_task(base_python()), capture_dir=capture_dir) as mock:
        # The public API: a config, ProxyRecorder(config), start(), run Codex, stop().
        config = {
            "upstream": {"name": "mock", "base_url": mock.base_url, "wire_api": "responses", "auth": {"mode": "none"}},
            "listen": {"host": "127.0.0.1", "port": 0},
            "log_dir": str(run_root / "codexprobe"),
            "seed": 0,
        }
        recorder = ProxyRecorder(config)
        endpoint = recorder.start()
        try:
            toml = codex_config(endpoint)
            config_path = write_codex_config(codex_home, toml)
            print(f"Codex config.toml, written to {config_path}:\n\n{toml}")
            command = exec_command_line(codex, workspace, run_root / "last-message.txt", CODING_TASK_PROMPT)
            print(f"Running {CODEX_VERSION}: codex exec -s workspace-write ... {CODING_TASK_PROMPT!r}", flush=True)
            exit_code, timed_out, seconds = run_codex(command, env, run_root, timeout)
            mock.wait_for_exchanges(len(mock.exchanges))  # every capture is on disk
        finally:
            calls = recorder.stop()

    print(f"\ncodex exec exited with {exit_code} after {seconds:.1f} s")
    print(f"\nLLM calls recorded by CodexProbe ({len(calls)}):")
    for line in summarize(calls):
        print(f"  {line}")
    print(f"Log directory: {recorder.run_dir}")
    print(f"Workspace: {workspace}")

    # Independent checks: not Codex's exit code or its final message.
    report = verify(
        RunEvidence(workspace, exit_code, capture_dir, recorder.run_dir),
        expected_files=CODING_TASK_FILES,
        python=sys.executable,
    )
    print(f"\n{report.text()}")
    if timed_out:
        print(f"FAIL: codex exec did not finish within {timeout:.0f} s and was killed")
        return FAIL
    return PASS if report.passed else FAIL


def run_qwen(
    codex: str,
    codex_home: Path,
    env: dict[str, str],
    timeout: float,
    run_root: Path,
    workspace: Path,
    ollama_url: str,
    task: str,
) -> int:
    """Codex -> CodexProbe -> Ollama's qwen2.5-coder:1.5b: the coding task, or with ``task="text"`` a text reply."""
    prompt = QWEN_CODE_PROMPT if task == "code" else QWEN_PROMPT
    # The public API, the same as in mock mode: only the upstream differs.
    config = {
        "upstream": {"name": "ollama", "base_url": ollama_url, "wire_api": "responses", "auth": {"mode": "none"}},
        "listen": {"host": "127.0.0.1", "port": 0},
        "log_dir": str(run_root / "codexprobe"),
        "seed": 0,
    }
    recorder = ProxyRecorder(config)
    endpoint = recorder.start()
    try:
        toml = qwen_codex_config(endpoint)
        config_path = write_codex_config(codex_home, toml)
        print(f"Codex config.toml, written to {config_path}:\n\n{toml}")
        command = exec_command_line(codex, workspace, run_root / "last-message.txt", prompt)
        print(f"Running {CODEX_VERSION} with {QWEN_MODEL}: codex exec -s workspace-write ... {prompt!r}", flush=True)
        exit_code, timed_out, seconds = run_codex(command, env, run_root, timeout)
    finally:
        calls = recorder.stop()

    last_message = run_root / "last-message.txt"
    print(f"\ncodex exec exited with {exit_code} after {seconds:.1f} s" + (" (killed at the time limit)" if timed_out else ""))
    if last_message.is_file():
        print(f"Codex's final message: {last_message.read_text('utf-8', errors='replace').strip()!r}"[:600])
    print(f"\nLLM calls recorded by CodexProbe ({len(calls)}):")
    for line in summarize(calls):
        print(f"  {line}")
    print(f"Upstream peer(s): {sorted({str(c['upstream']['peer']) for c in calls})}")
    print(f"Log directory: {recorder.run_dir}")
    print(f"Workspace: {workspace}")

    if task == "text":
        report = Report(tuple(text_reply_checks(calls, exit_code, timed_out, QWEN_REPLY)))
    else:
        events = codex_events(run_root / "codex-stdout.jsonl")
        print("\nWhat the model asked Codex to run:")
        for line in tool_call_lines(calls) or ["(no tool calls)"]:
            print(f"  {line}")
        print("What Codex reports it ran:")
        for line in execution_lines(events) or ["(nothing)"]:
            print(f"  {line}")
        report = Report(tuple(coding_facts(calls, events, workspace, sys.executable, backend_address(ollama_url))))
    print(f"\n{report.text()}")
    return PASS if report.passed else FAIL


def run_openai(codex: str, env: dict[str, str], timeout: float, run_root: Path, workspace: Path, task: str) -> int:
    """Codex, signed in with ChatGPT -> CodexProbe -> OpenAI's Codex backend: the coding task, or with ``task="text"`` a text reply."""
    prompt = CODING_TASK_PROMPT if task == "code" else OPENAI_PROMPT
    # The public API again. Codex's own credentials pass through unchanged ("passthrough").
    config = {
        "upstream": {"name": "openai-chatgpt", "base_url": OPENAI_UPSTREAM, "wire_api": "responses", "auth": {"mode": "passthrough"}},
        "listen": {"host": "127.0.0.1", "port": 0},
        "log_dir": str(run_root / "codexprobe"),
        "seed": 0,
    }
    recorder = ProxyRecorder(config)
    endpoint = recorder.start()
    try:
        overrides = openai_overrides(endpoint)
        print("Codex settings, passed as -c overrides (nothing is written to CODEX_HOME):")
        for override in overrides:
            print(f"  -c {override}")
        print(f"\nThe same settings as a config.toml fragment:\n\n{config_snippet(overrides)}")
        command = exec_command_line(codex, workspace, run_root / "last-message.txt", prompt, overrides)
        print(f"Running {CODEX_VERSION}, signed in with ChatGPT: codex exec -s workspace-write ... {prompt!r}", flush=True)
        exit_code, timed_out, seconds = run_codex(command, env, run_root, timeout)
    finally:
        calls = recorder.stop()

    last_message = run_root / "last-message.txt"
    print(f"\ncodex exec exited with {exit_code} after {seconds:.1f} s" + (" (killed at the time limit)" if timed_out else ""))
    if last_message.is_file():
        print(f"Codex's final message: {last_message.read_text('utf-8', errors='replace').strip()!r}"[:600])
    print(f"\nLLM calls recorded by CodexProbe ({len(calls)}):")
    for line in summarize(calls):
        print(f"  {line}")
    print(f"Upstream URL(s): {sorted({str(c['upstream']['url']) for c in calls})}")
    print(f"Upstream peer(s): {sorted({str(c['upstream']['peer']) for c in calls})}")
    print(f"Log directory: {recorder.run_dir}")
    print(f"Workspace: {workspace}")

    if task == "text":
        checks = openai_text_checks(calls, exit_code, timed_out)
    else:
        print("What Codex reports it ran:")
        for line in execution_lines(codex_events(run_root / "codex-stdout.jsonl")) or ["(nothing)"]:
            print(f"  {line}")
        checks = openai_checks(calls, workspace, sys.executable)
    # Unlike the mock, the real backend leaves no capture of its own to compare
    # with, so the log is checked against stop()'s list instead.
    report = Report((*checks, log_check(calls, recorder.run_dir)))
    print(f"\n{report.text()}")
    return PASS if report.passed else FAIL


if __name__ == "__main__":
    sys.exit(main())
