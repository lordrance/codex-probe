"""Run the real Codex CLI against one Responses endpoint: what the quickstart and the demo script share.

* :func:`codex_config` is the ``config.toml`` that sends Codex to one ``base_url``
  through a custom Responses provider. Retries, plugins, apps, analytics, web
  search and unbounded connection retries are off, and the Windows sandbox is
  named.
* :func:`write_codex_config` writes it into a CODEX_HOME. It refuses the
  default ``~/.codex`` and any ``config.toml`` that CodexProbe did not write.
* :func:`exec_command_line` and :func:`run_codex` run ``codex exec`` in a
  workspace: stdin closed, ``-s workspace-write``, a wall-clock limit, and the
  whole process tree killed on timeout.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CODEX_VERSION = "codex-cli 0.164.0"  # the version these runs were validated with
TIMEOUT_S = 600  # wall-clock limit for one codex exec against the mock
CONFIG_MARKER = "# Written by codex-probe"  # first line of every config.toml written here

# Credentials and proxy settings that must not reach the Codex run.
DROPPED_ENV = frozenset(
    {"CODEX_API_KEY", "CODEX_ACCESS_TOKEN", "OPENAI_API_KEY", "CODEX_SQLITE_HOME", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}
)


class CodexHomeError(RuntimeError):
    """The CODEX_HOME must not be used: it is the default one, or its config.toml was not written by CodexProbe."""


def codex_config(
    base_url: str,
    model: str = "codexprobe-mock",
    *,
    context_window: int | None = None,
    stream_idle_timeout_ms: int | None = None,
    minimal_tools: bool = False,
) -> str:
    """The ``config.toml`` for one run against ``base_url``.

    Codex does not retry, so every request the backend sees was sent once.
    Without ``[windows] sandbox``, Codex 0.164.0 selects no sandbox on Windows
    and ``codex exec`` rejects every command as "blocked by policy". The
    unelevated sandbox needs no administrator rights; the key is ignored
    elsewhere.

    For a local model:

    * ``context_window`` tells Codex the server's real context (Ollama's
      ``num_ctx``). Otherwise Codex assumes 272k for an unknown model.
    * ``stream_idle_timeout_ms`` allows for a slow prefill, during which Ollama
      sends no bytes at all.
    * ``minimal_tools`` leaves Codex only ``exec_command`` and ``write_stdin``,
      which makes every request much smaller.
    """
    lines = [
        f"{CONFIG_MARKER} (codex_probe.codex_run); rewritten before every run.",
        f'model = "{model}"',
        'model_provider = "codexprobe"',
        'web_search = "disabled"',
    ]
    if context_window is not None:
        lines.append(f"model_context_window = {context_window}")
    if minimal_tools:
        lines.append('model_reasoning_summary = "none"')
    lines += [
        "",
        "[model_providers.codexprobe]",
        'name = "CodexProbe"',
        f'base_url = "{base_url}"',
        'wire_api = "responses"',
        "request_max_retries = 0",
        "stream_max_retries = 0",
    ]
    if stream_idle_timeout_ms is not None:
        lines.append(f"stream_idle_timeout_ms = {stream_idle_timeout_ms}")
    lines += [
        "",
        "[features]",
        "unbounded_connection_retries = false",
        "plugins = false",
        "apps = false",
    ]
    if minimal_tools:
        lines += [
            "view_image = false",
            "tool_suggest = false",
            "goals = false",
            "",
            "[agents]",
            "enabled = false",
            "",
            "[tools.experimental_request_user_input]",
            "enabled = false",
            "",
            "[skills]",
            "include_instructions = false",
        ]
    lines += [
        "",
        "[analytics]",
        "enabled = false",
        "",
        "[windows]",
        'sandbox = "unelevated"',
        "",
    ]
    return "\n".join(lines)


def check_codex_home(codex_home: Path) -> None:
    """Raise :class:`CodexHomeError` unless CodexProbe may write ``config.toml`` into ``codex_home``."""
    if codex_home.resolve() == (Path.home() / ".codex").resolve():
        raise CodexHomeError(f"{codex_home} is the default CODEX_HOME; use a separate directory")
    target = codex_home / "config.toml"
    if target.exists() and not target.read_text("utf-8", errors="replace").startswith(CONFIG_MARKER):
        raise CodexHomeError(f"{target} was not written by CodexProbe, so it will not be replaced")


def write_codex_config(codex_home: Path, text: str) -> Path:
    """Write ``text`` as ``codex_home/config.toml`` after :func:`check_codex_home`; return the path."""
    check_codex_home(codex_home)
    codex_home.mkdir(parents=True, exist_ok=True)
    target = codex_home / "config.toml"
    target.write_text(text, "utf-8")
    return target


def codex_env(codex_home: Path) -> dict[str, str]:
    """This process's environment for Codex: CODEX_HOME set, credentials and proxies removed."""
    env = {key: value for key, value in os.environ.items() if key.upper() not in DROPPED_ENV}
    env["CODEX_HOME"] = str(codex_home)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    return env


def codex_version(codex: str | os.PathLike[str], env: dict[str, str]) -> str:
    """What ``codex --version`` prints, or "" if it cannot be run."""
    try:
        done = subprocess.run(
            [str(codex), "--version"], stdin=subprocess.DEVNULL, capture_output=True, text=True, env=env, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip()


def base_python() -> Path:
    """This Python's base interpreter, outside any virtual environment: Codex runs the tests with it."""
    if os.name == "nt":
        return Path(sys.base_prefix) / "python.exe"
    return Path(sys.base_prefix) / "bin" / f"python{sys.version_info[0]}.{sys.version_info[1]}"


def exec_command_line(
    codex: str | os.PathLike[str],
    workspace: Path,
    last_message: Path,
    prompt: str,
    overrides: Sequence[str] = (),
) -> list[str]:
    """``codex exec`` in ``workspace`` with the workspace-write sandbox and JSONL events on stdout.

    Each of ``overrides`` (``key=value``, the value in TOML) is passed with its own ``-c``.
    """
    return [
        str(codex), "exec", "--json", "--skip-git-repo-check", "--strict-config",
        "-s", "workspace-write", "-c", "allow_login_shell=false",
        *(part for override in overrides for part in ("-c", override)),
        "-C", str(workspace), "-o", str(last_message), prompt,
    ]  # fmt: skip


def run_codex(command: list[str], env: dict[str, str], out: Path, timeout: float) -> tuple[int, bool, float]:
    """Run codex exec with stdin closed. Return its exit code, whether it hit the
    timeout, and the seconds it took.

    Its stdout (the JSONL events) and stderr go to ``codex-stdout.jsonl`` and
    ``codex-stderr.txt`` in ``out``. On timeout the whole process tree is killed.
    """
    started = time.perf_counter()
    extra: dict[str, Any] = {} if os.name == "nt" else {"start_new_session": True}
    with open(out / "codex-stdout.jsonl", "wb") as stdout, open(out / "codex-stderr.txt", "wb") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=env, **extra)
        try:
            process.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            timed_out = True
    return process.returncode, timed_out, time.perf_counter() - started
