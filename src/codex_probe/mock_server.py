"""A scriptable mock of the OpenAI Responses API, for tests, the Phase 2 Codex runs and the quickstart.

The SSE follows the fixtures in Codex 0.164.0's own tests
(``codex-rs/core/tests/common/responses.rs``):

* each event is written as ``event: <type>``, then ``data: <compact JSON>``,
  then a blank line;
* tool calls carry their complete arguments in ``response.output_item.done``;
* ``response.completed`` carries a response id and a zeroed ``usage``.

Like those tests, ``GET /v1/models`` answers ``{"models": []}`` so a client
that asks stays hermetic, and ``GET /v1/responses`` (a WebSocket attempt)
gets 426 so the client falls back to HTTP.

The mock is the independent witness for CodexProbe's records. It keeps every
request it receives and every chunk it sends, byte for byte, and can save them
to a directory the caller chooses. In the saved files, credential header
values are redacted; bodies are kept exactly.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from codex_probe._server import AbortConnection, BackgroundServer, ServerStartError
from codex_probe.storage import redact_headers

ZERO_USAGE: dict[str, Any] = {
    "input_tokens": 0,
    "input_tokens_details": None,
    "output_tokens": 0,
    "output_tokens_details": None,
    "total_tokens": 0,
}
SSE_HEADERS: tuple[tuple[str, str], ...] = (
    ("content-type", "text/event-stream"),
    ("cache-control", "no-cache"),
)

# sent: the whole response went out; aborted: the connection was dropped on purpose
# (``end="abort"``); peer_gone: the client disconnected while the mock was waiting to
# send; cancelled: the server stopped first.
ExchangeState = Literal["in_progress", "sent", "aborted", "peer_gone", "cancelled"]


class MockServerError(RuntimeError):
    """The mock server could not start or stop."""


# --- event builders, as in Codex's core/tests/common/responses.rs ------------


def ev_response_created(response_id: str) -> dict[str, Any]:
    """``response.created`` with the given id."""
    return {"type": "response.created", "response": {"id": response_id}}


def ev_assistant_message(item_id: str, text: str) -> dict[str, Any]:
    """A complete assistant message in ``response.output_item.done``."""
    return {
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "role": "assistant",
            "id": item_id,
            "content": [{"type": "output_text", "text": text}],
        },
    }


def ev_function_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    """A complete function call in ``response.output_item.done``; ``arguments`` is a JSON string."""
    return {
        "type": "response.output_item.done",
        "item": {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments},
    }


def ev_exec_command_call(call_id: str, cmd: str) -> dict[str, Any]:
    """A call to ``exec_command``, the shell tool Codex 0.164.0 offers to unknown models."""
    return ev_function_call(call_id, "exec_command", _compact({"cmd": cmd}))


def ev_apply_patch_exec_command_call(call_id: str, patch: str) -> dict[str, Any]:
    """Edit files through ``exec_command``. Codex intercepts the ``apply_patch`` heredoc itself."""
    return ev_exec_command_call(call_id, f"apply_patch <<'EOF'\n{patch}\nEOF\n")


def ev_completed(response_id: str, usage: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``response.completed`` with an id and a usage object (zeroed by default)."""
    return {
        "type": "response.completed",
        "response": {"id": response_id, "usage": dict(usage) if usage is not None else dict(ZERO_USAGE)},
    }


def ev_failed(response_id: str, code: str, message: str) -> dict[str, Any]:
    """``response.failed`` with an error object."""
    return {
        "type": "response.failed",
        "response": {"id": response_id, "error": {"code": code, "message": message}},
    }


def sse(events: Iterable[Mapping[str, Any]]) -> bytes:
    """Encode events like Codex's ``sse()`` test helper.

    Each event becomes ``event: <type>``, then ``data: <json>``, then a blank
    line. An event that has only a ``type`` key gets no data line.
    """
    parts: list[str] = []
    for event in events:
        parts.append(f"event: {event['type']}\n")
        if set(event) != {"type"}:
            parts.append(f"data: {_compact(event)}\n")
        parts.append("\n")
    return "".join(parts).encode("utf-8")


# --- scripted responses ------------------------------------------------------


@dataclass(frozen=True)
class FunctionCall:
    """A tool call the mock sent. The next request must carry its ``function_call_output``."""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class MockResponse:
    """One scripted HTTP response, sent exactly as described."""

    status: int = 200
    headers: tuple[tuple[str, str], ...] = SSE_HEADERS
    chunks: tuple[bytes, ...] = ()  # body chunks, each handed to the server separately
    first_byte_delay: float = 0.0  # seconds to wait before the status line and headers
    chunk_delay: float = 0.0  # seconds to wait before each chunk after the first
    end: Literal["complete", "abort"] = "complete"  # "abort" drops the connection after the last chunk
    function_calls: tuple[FunctionCall, ...] = ()


def sse_response(
    events: Iterable[Mapping[str, Any]],
    *,
    first_byte_delay: float = 0.0,
    event_delay: float = 0.0,
    end: Literal["complete", "abort"] = "complete",
) -> MockResponse:
    """A 200 SSE response that sends one chunk per event.

    The function calls the next request must answer are taken from the events.
    To end the stream before a terminal event, leave the terminal event out
    of ``events``.
    """
    events = list(events)
    return MockResponse(
        chunks=tuple(sse([event]) for event in events),
        first_byte_delay=first_byte_delay,
        chunk_delay=event_delay,
        end=end,
        function_calls=_function_calls(events),
    )


def raw_response(
    chunks: Sequence[bytes],
    *,
    status: int = 200,
    headers: Sequence[tuple[str, str]] = SSE_HEADERS,
    first_byte_delay: float = 0.0,
    chunk_delay: float = 0.0,
    end: Literal["complete", "abort"] = "complete",
    function_calls: Sequence[FunctionCall] = (),
) -> MockResponse:
    """A response that sends exactly these chunks, which may split events anywhere."""
    return MockResponse(
        status=status,
        headers=tuple(headers),
        chunks=tuple(chunks),
        first_byte_delay=first_byte_delay,
        chunk_delay=chunk_delay,
        end=end,
        function_calls=tuple(function_calls),
    )


def json_response(status: int, body: Any, *, first_byte_delay: float = 0.0) -> MockResponse:
    """A complete JSON response with any status, e.g. an HTTP error from the backend."""
    data = _compact(body).encode("utf-8")
    return MockResponse(
        status=status,
        headers=(("content-type", "application/json"), ("content-length", str(len(data)))),
        chunks=(data,),
        first_byte_delay=first_byte_delay,
    )


# --- the coding task of the real Codex runs and the quickstart ---------------

CODING_TASK_PROMPT = "Write calc.py with add(a, b) and a unittest test for it in test_calc.py, run the tests, and report."
CODING_TASK_FILES = ("calc.py", "test_calc.py")
CODING_TASK_ANSWER = "Wrote calc.py and test_calc.py; the unittest run passed."
_CODING_TASK_PATCH = "\n".join(
    [
        "*** Begin Patch",
        "*** Add File: calc.py",
        "+def add(a, b):",
        "+    return a + b",
        "*** Add File: test_calc.py",
        "+import unittest",
        "+",
        "+from calc import add",
        "+",
        "+",
        "+class AddTest(unittest.TestCase):",
        "+    def test_add(self):",
        "+        self.assertEqual(add(2, 3), 5)",
        "+",
        "+",
        '+if __name__ == "__main__":',
        "+    unittest.main()",
        "*** End Patch",
    ]
)


def unittest_command(python: str | os.PathLike[str], *, windows: bool = os.name == "nt") -> str:
    """The shell command that runs the workspace's tests with ``python``.

    Codex runs ``exec_command`` in PowerShell on Windows and in a POSIX shell
    elsewhere, so the quoting differs. ``-B`` keeps ``__pycache__`` out of the
    workspace: a ``.pyc`` records its source's modification time and path, so two
    runs would leave different bytes.
    """
    if windows:
        quoted = "'" + str(python).replace("'", "''") + "'"
        return f"& {quoted} -B -m unittest -v"
    return f"{shlex.quote(str(python))} -B -m unittest -v"


def coding_task(python: str | os.PathLike[str]) -> list[MockResponse]:
    """The Phase 2 coding task, in three scripted turns.

    1. One ``exec_command`` with an ``apply_patch`` heredoc (Codex intercepts it)
       that adds ``calc.py`` and its ``unittest`` test, ``test_calc.py``.
    2. One ``exec_command`` that runs the tests with ``python``. It waits up to
       30 s for the result, which leaves room for the shell and Python to start
       in Codex's sandbox.
    3. The final answer, :data:`CODING_TASK_ANSWER`.

    The mock answers each turn only if the request carries the previous turn's
    tool output.
    """
    run_tests = _compact({"cmd": unittest_command(python), "yield_time_ms": 30000})
    return [
        sse_response(
            [
                ev_response_created("resp-1"),
                ev_apply_patch_exec_command_call("call-1", _CODING_TASK_PATCH),
                ev_completed("resp-1"),
            ]
        ),
        sse_response(
            [ev_response_created("resp-2"), ev_function_call("call-2", "exec_command", run_tests), ev_completed("resp-2")]
        ),
        sse_response(
            [ev_response_created("resp-3"), ev_assistant_message("msg-3", CODING_TASK_ANSWER), ev_completed("resp-3")]
        ),
    ]


# --- what the mock saw and sent ----------------------------------------------


@dataclass
class Exchange:
    """One request the mock received and the response it sent, byte for byte.

    ``response_chunks`` are the chunks the app handed to the server, in order.
    """

    index: int
    method: str
    path: str  # the raw request path, without the query
    query: str
    request_headers: list[tuple[str, str]]
    request_body: bytes
    status: int | None = None
    response_headers: list[tuple[str, str]] = field(default_factory=list)
    response_chunks: list[bytes] = field(default_factory=list)
    response_chunk_times: list[float] = field(default_factory=list)  # time.perf_counter() as each chunk was sent
    step: int | None = None  # index of the scripted response that answered, if any
    function_calls: tuple[FunctionCall, ...] = ()  # tool calls this response issued
    function_call_outputs: dict[str, Any] = field(default_factory=dict)  # call_id -> output, from the request
    failure: str | None = None  # why the mock refused the request, if it did
    state: ExchangeState = "in_progress"

    @property
    def response_body(self) -> bytes:
        """All response chunks joined: the complete body the mock sent."""
        return b"".join(self.response_chunks)


def load_capture(directory: str | os.PathLike[str]) -> list[Exchange]:
    """Read back the exchanges a :class:`MockResponsesServer` saved with ``capture_dir``.

    Raises:
        OSError: a file of the capture is missing or unreadable.
        ValueError: a file is not valid JSON, or a body file is not the size the
            capture recorded (cut short or padded).
    """
    root = Path(directory)
    exchanges: list[Exchange] = []
    for meta_path in sorted(root.glob("*-exchange.json")):
        meta = json.loads(meta_path.read_text("utf-8"))
        prefix = meta_path.name.removesuffix("-exchange.json")
        request_body = (root / f"{prefix}-request.bin").read_bytes()
        response_body = (root / f"{prefix}-response.bin").read_bytes()
        if len(request_body) != meta["request_size"]:
            raise ValueError(
                f"{prefix}-request.bin has {len(request_body)} bytes; the capture recorded {meta['request_size']}"
            )
        if len(response_body) != sum(meta["response_chunk_sizes"]):
            raise ValueError(
                f"{prefix}-response.bin has {len(response_body)} bytes; "
                f"the capture recorded {sum(meta['response_chunk_sizes'])}"
            )
        chunks: list[bytes] = []
        offset = 0
        for size in meta["response_chunk_sizes"]:
            chunks.append(response_body[offset : offset + size])
            offset += size
        exchanges.append(
            Exchange(
                index=meta["index"],
                method=meta["method"],
                path=meta["path"],
                query=meta["query"],
                request_headers=[(name, value) for name, value in meta["request_headers"]],
                request_body=request_body,
                status=meta["status"],
                response_headers=[(name, value) for name, value in meta["response_headers"]],
                response_chunks=chunks,
                response_chunk_times=meta["response_chunk_times"],
                step=meta["step"],
                function_calls=tuple(FunctionCall(**call) for call in meta["function_calls"]),
                function_call_outputs=meta["function_call_outputs"],
                failure=meta["failure"],
                state=meta["state"],
            )
        )
    return exchanges


# --- the server --------------------------------------------------------------


class MockResponsesServer:
    """Serve scripted Responses API answers on 127.0.0.1 and capture every exchange.

    Each ``POST /v1/responses`` takes the next scripted response. If the
    previous response issued function calls, the request must carry a
    ``function_call_output`` for each of their ``call_id``s. Otherwise, or when
    the script has run out, the mock answers 400 with a ``mock_script_error``
    and records the reason in :attr:`failures`.
    """

    def __init__(
        self,
        script: Iterable[MockResponse] = (),
        *,
        port: int = 0,
        capture_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self._script: list[MockResponse] = list(script)
        self._next_step = 0
        self._pending: tuple[FunctionCall, ...] = ()
        self._pending_step: int | None = None
        self._capture_dir = Path(capture_dir) if capture_dir is not None else None
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._exchanges: list[Exchange] = []
        self._background = BackgroundServer(
            self._app, port=port, lifespan="off", graceful_timeout=2, thread_name="codexprobe-mock"
        )
        self.base_url: str | None = None
        """``http://127.0.0.1:<port>/v1`` while running, else None."""
        self.capture_errors: list[str] = []
        """Problems saving captures to ``capture_dir``. Serving goes on regardless."""

    # lifecycle

    def start(self, timeout: float = 10.0) -> str:
        """Start serving and return the base URL, e.g. ``http://127.0.0.1:8135/v1``.

        Raises:
            MockServerError: already running, the port cannot be bound, or the
                server did not come up within ``timeout`` seconds.
        """
        if self._capture_dir is not None:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
        try:
            port = self._background.start(timeout)
        except ServerStartError as exc:
            raise MockServerError(f"mock server: {exc}") from exc
        self.base_url = f"http://127.0.0.1:{port}/v1"
        return self.base_url

    def stop(self, timeout: float = 10.0) -> None:
        """Stop serving and free the port. Calling it again, or before start(), does nothing.

        A response still being sent is cancelled after a 2-second grace period.
        If the server has not stopped after ``timeout`` seconds, it is forced
        to exit.
        """
        self.base_url = None
        try:
            self._background.stop(timeout)
        except RuntimeError as exc:
            raise MockServerError(str(exc)) from exc

    def __enter__(self) -> MockResponsesServer:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # script and results

    def add(self, *responses: MockResponse) -> None:
        """Append responses to the script. Allowed while the server is running."""
        with self._lock:
            self._script.extend(responses)

    @property
    def exchanges(self) -> list[Exchange]:
        """Every exchange so far, in arrival order."""
        with self._lock:
            return list(self._exchanges)

    @property
    def failures(self) -> list[str]:
        """Why the mock refused requests: missing tool outputs, an exhausted script, unknown routes."""
        return [exchange.failure for exchange in self.exchanges if exchange.failure is not None]

    def wait_for_exchanges(self, count: int, timeout: float = 10.0) -> list[Exchange]:
        """Wait until ``count`` exchanges are no longer in progress; return all exchanges."""
        deadline = time.monotonic() + timeout
        with self._changed:
            while sum(e.state != "in_progress" for e in self._exchanges) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"{count} exchange(s) did not finish within {timeout} s")
                self._changed.wait(remaining)
            return list(self._exchanges)

    # the ASGI application

    async def _app(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        exchange = Exchange(
            index=-1,
            method=scope["method"],
            path=scope.get("raw_path", scope["path"].encode()).decode("latin-1"),
            query=scope["query_string"].decode("latin-1"),
            request_headers=[(k.decode("latin-1"), v.decode("latin-1")) for k, v in scope["headers"]],
            request_body=bytes(body),
        )
        with self._lock:
            exchange.index = len(self._exchanges)
            self._exchanges.append(exchange)
        disconnected = asyncio.Event()
        watcher = asyncio.create_task(_watch_disconnect(receive, disconnected))
        final_state: ExchangeState = "cancelled"  # unless the response is sent or ends otherwise
        try:
            response = self._choose_response(exchange, scope["path"])
            await self._send(response, exchange, send, disconnected)
            final_state = "sent"
        except _PeerGone:
            final_state = "peer_gone"
        except AbortConnection:
            final_state = "aborted"
            raise
        finally:
            watcher.cancel()
            # Save first, then publish the final state: a waiter never sees a
            # finished exchange whose files are not written yet.
            self._save(exchange, final_state)
            with self._changed:
                exchange.state = final_state
                self._changed.notify_all()

    def _choose_response(self, exchange: Exchange, path: str) -> MockResponse:
        if exchange.method == "GET" and path == "/v1/models":
            return json_response(200, {"models": []})
        if exchange.method == "GET" and path == "/v1/responses":
            return json_response(426, {"error": {"message": "websockets unsupported"}})
        if exchange.method != "POST" or path != "/v1/responses":
            return self._refuse(exchange, 404, f"no mock route for {exchange.method} {path}")
        request = _json_or_none(exchange.request_body)
        exchange.function_call_outputs = _function_call_outputs(request)
        with self._lock:
            missing = [c.call_id for c in self._pending if c.call_id not in exchange.function_call_outputs]
            if missing:
                return self._refuse(
                    exchange,
                    400,
                    f"request {exchange.index} has no function_call_output for call_id(s) "
                    f"{', '.join(missing)} issued by script step {self._pending_step}",
                )
            if self._next_step >= len(self._script):
                return self._refuse(
                    exchange,
                    400,
                    f"mock script exhausted: {len(self._script)} response(s) were scripted "
                    f"and all have been used; request {exchange.index} has no answer",
                )
            step = self._next_step
            self._next_step += 1
            response = self._script[step]
            self._pending, self._pending_step = response.function_calls, step
        exchange.step = step
        exchange.function_calls = response.function_calls
        return response

    def _refuse(self, exchange: Exchange, status: int, reason: str) -> MockResponse:
        exchange.failure = reason
        return json_response(
            status,
            {"error": {"type": "invalid_request_error", "code": "mock_script_error", "message": reason}},
        )

    async def _send(
        self, response: MockResponse, exchange: Exchange, send: Any, disconnected: asyncio.Event
    ) -> None:
        if response.first_byte_delay and await _peer_left_during(response.first_byte_delay, disconnected):
            raise _PeerGone
        await send(
            {
                "type": "http.response.start",
                "status": response.status,
                "headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in response.headers],
            }
        )
        exchange.status = response.status
        exchange.response_headers = list(response.headers)
        for index, chunk in enumerate(response.chunks):
            if index and response.chunk_delay and await _peer_left_during(response.chunk_delay, disconnected):
                raise _PeerGone
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
            exchange.response_chunk_times.append(time.perf_counter())
            exchange.response_chunks.append(chunk)
        if response.end == "abort":
            raise AbortConnection  # uvicorn closes the connection without ending the body
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    def _save(self, exchange: Exchange, state: ExchangeState) -> None:
        if self._capture_dir is None:
            return
        prefix = self._capture_dir / f"{exchange.index:04d}"
        meta = {
            "index": exchange.index,
            "method": exchange.method,
            "path": exchange.path,
            "query": exchange.query,
            "request_headers": redact_headers(exchange.request_headers),
            "request_size": len(exchange.request_body),
            "status": exchange.status,
            "response_headers": redact_headers(exchange.response_headers),
            "response_chunk_sizes": [len(chunk) for chunk in exchange.response_chunks],
            "response_chunk_times": exchange.response_chunk_times,
            "step": exchange.step,
            "function_calls": [vars(call) for call in exchange.function_calls],
            "function_call_outputs": exchange.function_call_outputs,
            "failure": exchange.failure,
            "state": state,
        }
        try:
            Path(f"{prefix}-request.bin").write_bytes(exchange.request_body)
            Path(f"{prefix}-response.bin").write_bytes(exchange.response_body)
            Path(f"{prefix}-exchange.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except (OSError, TypeError, ValueError) as exc:
            self.capture_errors.append(f"exchange {exchange.index}: {type(exc).__name__}: {exc}")


# --- helpers -----------------------------------------------------------------


class _PeerGone(Exception):
    """The client disconnected while the mock was waiting to send."""


async def _watch_disconnect(receive: Any, disconnected: asyncio.Event) -> None:
    """Set ``disconnected`` when the client goes away (call after the body was read)."""
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            disconnected.set()
            return


async def _peer_left_during(delay: float, disconnected: asyncio.Event) -> bool:
    """Wait ``delay`` seconds, but return True at once if the client disconnects."""
    try:
        await asyncio.wait_for(disconnected.wait(), timeout=delay)
    except TimeoutError:
        return False
    return True


def _compact(value: Any) -> str:
    """JSON the way serde_json writes it: no spaces, non-ASCII kept as is."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _function_calls(events: Sequence[Mapping[str, Any]]) -> tuple[FunctionCall, ...]:
    calls = []
    for event in events:
        item = event.get("item")
        if event.get("type") == "response.output_item.done" and isinstance(item, Mapping):
            if item.get("type") == "function_call":
                calls.append(FunctionCall(str(item["call_id"]), str(item["name"]), str(item["arguments"])))
    return tuple(calls)


def _json_or_none(data: bytes) -> Any:
    try:
        return json.loads(data)
    except ValueError:
        return None


def _function_call_outputs(request: Any) -> dict[str, Any]:
    """Map call_id to output for every ``function_call_output`` item in the request's ``input``."""
    outputs: dict[str, Any] = {}
    items = request.get("input") if isinstance(request, dict) else None
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            call_id = item.get("call_id")
            if isinstance(call_id, str):
                outputs[call_id] = item.get("output")
    return outputs
