"""Check that a local model's tool call runs through CodexProbe: a one-file task.

The real Codex CLI 0.164.0 runs through ``ProxyRecorder`` to a model served by Ollama,
and is asked to create ``hello.txt`` holding one fixed line. The task needs one tool
call; it is not a coding task. The run is judged by:

* five of the quickstart's checks: Backend Reached, Response Captured, Model Tool Call
  (a structured ``exec_command`` call), Arguments Valid (against the tool's schema in
  the same request), and Tool Executed (Codex ran it and sent the output back);
* the file's text, decoded by its byte-order mark, because Windows PowerShell 5.1's
  ``>`` writes UTF-16;
* the log read back from disk, which must equal the list ``stop()`` returned;
* a time limit: ``codex exec`` must end within ``--timeout`` seconds (default 900).

The models used in VALIDATION.md, built from ``examples/ollama``::

    ollama pull qwen2.5-coder:1.5b
    ollama create codexprobe-qwen2.5-coder-tools:1.5b -f examples/ollama/Modelfile.tools
    ollama pull qwen3.5:2b-q4_K_M
    ollama create codexprobe-qwen3.5:2b -f examples/ollama/Modelfile.qwen35

From the repository root, with ``ollama serve`` running::

    python examples/tool_call_check.py --model codexprobe-qwen2.5-coder-tools:1.5b --codex <codex.exe> --codex-home <dir>

The Codex settings are the quickstart's qwen settings with this model. Exit codes are
the quickstart's: 0 PASS, 1 FAIL, 2 bad arguments, 3 BLOCKED.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quickstart  # noqa: E402  the quickstart's checks and qwen settings
from codex_probe import ProxyRecorder  # noqa: E402
from codex_probe.codex_run import (  # noqa: E402
    CODEX_VERSION,
    CodexHomeError,
    check_codex_home,
    codex_config,
    codex_env,
    codex_version,
    exec_command_line,
    run_codex,
    write_codex_config,
)
from codex_probe.e2e_validation import Check, Report  # noqa: E402

TEXT = "QWEN_TOOL_OK"
PROMPT = f"Use the exec_command tool to create a file named hello.txt that contains exactly {TEXT}."
TIMEOUT_S = 900
MODELFILE_HINT = (
    "build it from examples/ollama/Modelfile.tools (Qwen2.5-Coder 1.5B) or examples/ollama/Modelfile.qwen35"
    " (Qwen3.5 2B), with the commands at the top of examples/tool_call_check.py"
)


def decode_text(content: bytes) -> tuple[str, str]:
    """Decode by byte-order mark; without one, as UTF-8. Bad bytes never raise."""
    for bom, encoding in ((b"\xef\xbb\xbf", "utf-8-sig"), (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16")):
        if content.startswith(bom):
            return content.decode(encoding, errors="replace"), encoding
    return content.decode("utf-8", errors="replace"), "utf-8"


def file_check(workspace: Path) -> Check:
    """hello.txt exists and holds TEXT, apart from surrounding whitespace."""
    target = workspace / "hello.txt"
    if not target.is_file():
        return Check("hello.txt", False, "hello.txt was not created")
    content = target.read_bytes()
    text, encoding = decode_text(content)
    return Check("hello.txt", text.strip() == TEXT, f"text {ascii(text)} ({encoding}, {len(content)} bytes)")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="the Ollama model, which must set num_ctx 16384")
    parser.add_argument("--codex", help="the Codex CLI to run (default: codex on PATH)")
    parser.add_argument("--codex-home", type=Path, help="CODEX_HOME for the run; config.toml is written there (default: a new temporary directory)")
    parser.add_argument("--log-dir", type=Path, default=quickstart.DEFAULT_LOG_DIR, help="where the log and the workspace go (default: %(default)s)")
    parser.add_argument("--timeout", type=quickstart._seconds, default=TIMEOUT_S, help="seconds before codex exec is killed (default: %(default)s)")
    parser.add_argument("--ollama-url", type=quickstart._ollama_url, default=quickstart.QWEN_URL, help="Ollama's API, ending in /v1 (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    codex = shutil.which(args.codex or "codex")
    if codex is None:
        return quickstart.blocked(f"the Codex CLI was not found ({args.codex or 'codex on PATH'}); install @openai/codex@0.164.0")
    codex_home = args.codex_home or Path(tempfile.mkdtemp(prefix="codexprobe-codex-home-"))
    try:
        check_codex_home(codex_home)
    except CodexHomeError as exc:
        return quickstart.blocked(str(exc))
    problem = quickstart.ollama_problem(args.ollama_url, args.model)
    if problem is not None:
        if "examples/ollama/Modelfile" in problem:  # the quickstart's hint names its own Modelfile
            problem = f"{problem.partition('; ')[0]}; {MODELFILE_HINT}"
        return quickstart.blocked(problem)
    codex_home.mkdir(parents=True, exist_ok=True)
    env = codex_env(codex_home)
    version = codex_version(codex, env)
    if version != CODEX_VERSION:
        return quickstart.blocked(f"{codex} reports {version!r}; this check needs {CODEX_VERSION!r}")

    run_root = args.log_dir / f"toolcall-{datetime.now():%Y%m%d-%H%M%S}"
    workspace = run_root / "workspace"
    workspace.mkdir(parents=True)  # new and empty: anything in it afterwards came from Codex
    config = {
        "upstream": {"name": "ollama", "base_url": args.ollama_url, "wire_api": "responses", "auth": {"mode": "none"}},
        "listen": {"host": "127.0.0.1", "port": 0},
        "log_dir": str(run_root / "codexprobe"),
        "seed": 0,
    }
    recorder = ProxyRecorder(config)
    endpoint = recorder.start()
    try:
        toml = codex_config(
            endpoint,
            model=args.model,
            context_window=quickstart.QWEN_CONTEXT,
            stream_idle_timeout_ms=quickstart.QWEN_IDLE_TIMEOUT_MS,
            minimal_tools=True,
        )
        config_path = write_codex_config(codex_home, toml)
        print(f"Codex config.toml, written to {config_path}:\n\n{toml}")
        command = exec_command_line(codex, workspace, run_root / "last-message.txt", PROMPT)
        print(f"Running {CODEX_VERSION} with {args.model}: codex exec -s workspace-write ... {PROMPT!r}", flush=True)
        exit_code, timed_out, seconds = run_codex(command, env, run_root, args.timeout)
    finally:
        calls = recorder.stop()

    events = quickstart.codex_events(run_root / "codex-stdout.jsonl")
    print(f"\ncodex exec exited with {exit_code} after {seconds:.1f} s" + (" (killed at the time limit)" if timed_out else ""))
    print(f"\nLLM calls recorded by CodexProbe ({len(calls)}):")
    for line in quickstart.summarize(calls):
        print(f"  {line}")
    print(f"Log directory: {recorder.run_dir}")
    print("What the model asked Codex to run:")
    for line in quickstart.tool_call_lines(calls) or ["(no tool calls)"]:
        print(f"  {line}")
    print("What Codex reports it ran:")
    for line in quickstart.execution_lines(events) or ["(nothing)"]:
        print(f"  {line}")

    # The sixth fact, Coding Task, belongs to the quickstart's coding task and is dropped. It is given
    # a folder that does not exist, so it never runs the tests of files the model may have written.
    no_coding_task = run_root / "no-coding-task"
    facts = quickstart.coding_facts(calls, events, no_coding_task, sys.executable, quickstart.backend_address(args.ollama_url))
    checks = [
        *facts[:5],
        file_check(workspace),
        quickstart.log_check(calls, recorder.run_dir),
        Check("Finished in time", not timed_out, f"codex exec ended after {seconds:.1f} s; limit {args.timeout:.0f} s"),
    ]
    report = Report(tuple(checks))
    print(f"\n{report.text()}")
    return quickstart.PASS if report.passed else quickstart.FAIL


if __name__ == "__main__":
    # As in the quickstart: characters that the output's code page lacks are escaped.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="backslashreplace")
    sys.exit(main())
