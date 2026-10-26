"""The transparent recording proxy: :class:`ProxyRecorder`.

Codex is pointed at ``http://127.0.0.1:<port>/v1``. Every request under
``/v1/`` is forwarded to ``upstream.base_url`` plus the rest of the path, with
the query string and the body unchanged. The upstream response is relayed back
unchanged, chunk by chunk, as it arrives. Each ``POST /v1/responses`` becomes
one call record. Other requests, such as ``GET /v1/models``, are forwarded and
logged to ``other-requests.jsonl``, but they are not calls.

The only changes made, each listed in the record's ``upstream.changes``:

* hop-by-hop headers, and any header named in ``Connection``, are dropped in
  both directions;
* ``Host`` is set to the upstream host;
* credentials follow ``upstream.auth.mode``: kept (``passthrough``), dropped
  (``none``), or replaced by ``Authorization: Bearer $<env_var>`` (``env``).

Header names arrive lower-cased, because the HTTP server (h11) normalises them;
HTTP header names are case-insensitive. The proxy never adds headers such as
User-Agent, Accept-Encoding, Via, X-Forwarded-*, Date or Server. It never
retries either: retrying is Codex's job.

A record's ``response`` is the HTTP response the client actually received. It
is either the upstream's (``source: "upstream"``) or one the proxy generated
itself (``source: "proxy"``), such as a 502 when the upstream cannot be
reached. All its fields are null only when no response was produced at all,
for example when the client left before the upstream answered.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import threading
import time
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from codex_probe._server import AbortConnection, BackgroundServer, ServerStartError
from codex_probe.config import ConfigError, ProbeConfig, load_config
from codex_probe.storage import (
    LogWriteError,
    RecordError,
    RunWriter,
    call_identity,
    encode_body,
    is_sensitive_header,
    redact_headers,
)
from codex_probe.streaming import ParsedEvent, ResponseStreamAssembler, StreamEnd, StreamResult

# No read timeout: a local model can be silent for minutes while it reads the
# prompt. Only Codex's stream_idle_timeout_ms decides that a stream is stuck,
# so the proxy is never the first to cut it.
UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=None)
# After the client hangs up at a terminal event, keep reading the upstream this long.
DRAIN_AFTER_TERMINAL_S = 2.0
# stop() lets calls that are about to finish do so before it interrupts the rest.
STOP_GRACE_S = 1.0
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "proxy-authenticate",
        "proxy-authorization",
        "http2-settings",
    }
)

RawHeaders = list[tuple[bytes, bytes]]

# How a relay ended -> the end reported to the stream parser.
_STREAM_END: dict[str, StreamEnd] = {
    "eof": "eof",
    "upstream_error": "eof",
    "client_disconnected": "client_disconnected",
    "recorder_stopped": "recorder_stopped",
}
# How a relay ended -> the outcome of a response that is not an SSE stream.
_PLAIN_OUTCOME = {
    "eof": "completed",
    "upstream_error": "stream_cut",
    "client_disconnected": "client_disconnected",
    "recorder_stopped": "recorder_stopped",
}


class RecorderError(RuntimeError):
    """ProxyRecorder could not start or stop, or was used out of order."""


@dataclasses.dataclass
class _Call:
    """Everything observed about one forwarded request; it becomes a record at the end."""

    started_at: str
    arrival: float
    method: str
    path: str
    query: str
    headers: RawHeaders
    body: bytes
    is_llm: bool
    seq: int = -1
    call_id: str = ""
    request_index: int = -1
    url: str | None = None
    changes: list[str] = dataclasses.field(default_factory=list)
    status: int | None = None
    response_headers: RawHeaders | None = None
    response_started: bool = False
    peer: str | None = None
    first_byte: float | None = None
    first_chunk: float | None = None
    chunks: list[bytes] = dataclasses.field(default_factory=list)
    assembler: ResponseStreamAssembler | None = None
    end: str = "eof"  # eof | upstream_error | client_disconnected | recorder_stopped
    upstream_error: str | None = None
    client_error: str | None = None
    # Set when the proxy answered by itself instead of relaying the upstream.
    proxy_status: int | None = None
    proxy_headers: RawHeaders | None = None
    proxy_body: bytes | None = None
    proxy_reason: str | None = None

    def take(self, chunk: bytes) -> None:
        """Keep one upstream chunk and feed it to the stream parser."""
        if self.first_chunk is None:
            self.first_chunk = time.perf_counter()
        self.chunks.append(chunk)
        if self.assembler is not None:
            self.assembler.feed(chunk)


class ProxyRecorder:
    """Record every LLM call Codex makes, through a transparent local proxy.

    Usage::

        recorder = ProxyRecorder(config)
        endpoint = recorder.start()   # e.g. "http://127.0.0.1:8135/v1"
        # point model_providers.<id>.base_url at endpoint, run Codex
        calls = recorder.stop()       # one dict per LLM call, ordered by seq

    ``config`` is a dict, a path to a JSON file, or a :class:`ProbeConfig`
    (see :mod:`codex_probe.config`). One ``start()`` creates one run directory
    under ``log_dir``.
    """

    def __init__(self, config: ProbeConfig | Mapping[str, Any] | str | os.PathLike[str]) -> None:
        self.config = config if isinstance(config, ProbeConfig) else load_config(config)
        self._lock = threading.Lock()
        self._next_seq = 0
        self._next_request_index = 0
        self._server: BackgroundServer | None = None
        self._writer: RunWriter | None = None
        self._run_dir: Path | None = None
        self._endpoint: str | None = None
        self._started_at: str | None = None
        self._auth_value: str | None = None
        self._stopping = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: httpx.AsyncClient | None = None
        self._inflight: set[asyncio.Task[Any]] = set()
        self._submit_errors: list[str] = []
        self._result: list[dict[str, Any]] | None = None

    @property
    def endpoint(self) -> str | None:
        """The ``/v1`` URL Codex should use as its provider ``base_url``; None before start()."""
        return self._endpoint

    @property
    def run_dir(self) -> Path:
        """This run's log directory, as passed to :func:`codex_probe.load_session`."""
        if self._run_dir is None:
            raise RecorderError("run_dir is known only after start()")
        return self._run_dir

    def start(self) -> str:
        """Start the proxy on 127.0.0.1 and return the endpoint for Codex's provider ``base_url``.

        Raises:
            ConfigError: auth mode is "env" and its environment variable is unset or empty.
            RecorderError: already started, or the port could not be bound.
        """
        if self._server is not None:
            raise RecorderError("start() can be called only once per ProxyRecorder")
        auth = self.config.upstream.auth
        if auth.mode == "env":
            value = os.environ.get(auth.env_var or "", "")
            if not value:
                raise ConfigError(
                    f"upstream.auth.env_var: the environment variable {auth.env_var} is not set or is empty"
                )
            self._auth_value = value
        self._started_at = _utc_now()
        self._run_dir = self.config.log_dir / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"
        writer = RunWriter(self._run_dir)  # nothing is written until the first record
        server = BackgroundServer(
            self._app,
            port=self.config.listen.port,
            lifespan="on",
            graceful_timeout=5,
            thread_name="codexprobe-proxy",
        )
        try:
            port = server.start()
        except ServerStartError as exc:
            writer.stop()
            raise RecorderError(f"the proxy could not start: {exc}") from exc
        self._writer, self._server = writer, server
        self._endpoint = f"http://127.0.0.1:{port}/v1"
        writer.write_metadata(self._metadata())
        return self._endpoint

    def stop(self) -> list[dict[str, Any]]:
        """Stop the proxy, finish writing the logs, and return the LLM calls ordered by ``seq``.

        Calls still in flight are interrupted and recorded as ``recorder_stopped``.
        Stopping again returns the same list; stopping before start() returns [].

        Raises:
            LogWriteError: a record could not be written. The exception carries the calls.
            RecorderError: the server thread did not stop (the logs are still finished first).
        """
        if self._result is not None:
            return self._result
        if self._server is None or self._writer is None:
            return []
        self._stopping = True  # from now on, new requests are refused with a 503
        deadline = time.monotonic() + STOP_GRACE_S
        while self._inflight and time.monotonic() < deadline:
            time.sleep(0.01)  # a call whose reply was just delivered is usually done within ms
        loop = self._loop
        if loop is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):  # the loop may have closed in the meantime
                loop.call_soon_threadsafe(self._cancel_inflight)
        server_error: RuntimeError | None = None
        try:
            self._server.stop(timeout=20)
        except RuntimeError as exc:
            server_error = exc
        with self._lock:
            counts = {"llm_calls": self._next_seq, "other_requests": self._next_request_index}
        self._writer.write_metadata(self._metadata(finished_at=_utc_now(), **counts))
        try:
            calls = self._writer.stop()
        except LogWriteError as exc:
            self._result = exc.calls
            raise LogWriteError(str(exc), exc.calls, exc.errors + self._submit_errors) from None
        self._result = calls
        if self._submit_errors:
            raise LogWriteError(
                f"{len(self._submit_errors)} record(s) could not be logged; first: {self._submit_errors[0]}",
                calls,
                list(self._submit_errors),
            )
        if server_error is not None:
            raise RecorderError(f"the proxy server did not stop cleanly: {server_error}") from server_error
        return calls

    # --- ASGI ----------------------------------------------------------------

    async def _app(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
        elif scope["type"] == "http":
            task = asyncio.current_task()
            if task is not None:
                self._inflight.add(task)
            try:
                await self._handle(scope, receive, send)
            finally:
                if task is not None:
                    self._inflight.discard(task)

    async def _lifespan(self, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                self._loop = asyncio.get_running_loop()
                self._client = httpx.AsyncClient(
                    trust_env=False, timeout=UPSTREAM_TIMEOUT, follow_redirects=False
                )
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self._client is not None:
                    await self._client.aclose()
                await send({"type": "lifespan.shutdown.complete"})
                return

    def _cancel_inflight(self) -> None:
        for task in list(self._inflight):
            task.cancel()

    async def _handle(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        started_at, arrival = _utc_now(), time.perf_counter()
        body = await _read_body(receive)
        if body is None:
            return  # the client left before its request was complete; nothing was forwarded
        raw_path: bytes = scope.get("raw_path") or scope["path"].encode()
        call = _Call(
            started_at=started_at,
            arrival=arrival,
            method=scope["method"],
            path=raw_path.decode("latin-1"),
            query=scope["query_string"].decode("latin-1"),
            headers=list(scope["headers"]),
            body=body,
            is_llm=scope["method"] == "POST" and scope["path"] == "/v1/responses",
        )
        self._number(call)
        try:
            if not scope["path"].startswith("/v1/"):
                await self._proxy_error(send, call, 404, "CodexProbe forwards only paths under /v1/")
                return
            call.url = self.config.upstream.base_url + call.path[len("/v1") :]
            if call.query:
                call.url += "?" + call.query
            if self._stopping:
                await self._proxy_error(send, call, 503, "CodexProbe is stopping; the request was not forwarded")
                return
            await self._forward(call, receive, send)
        except asyncio.CancelledError:
            # stop() cancels the calls still in flight; uvicorn does the same once its
            # shutdown grace period is over. Record what was seen, then end the reply.
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
            call.end = "recorder_stopped"
            if not call.response_started:
                await self._proxy_error(send, call, 503, "CodexProbe stopped before the upstream answered")
            else:
                raise AbortConnection from None  # the body was cut: do not end it cleanly
        finally:
            self._finish(call)

    def _number(self, call: _Call) -> None:
        with self._lock:
            if call.is_llm:
                call.seq = self._next_seq
                self._next_seq += 1
                digest = hashlib.sha256(f"codexprobe:{self.config.seed}:{call.seq}".encode()).hexdigest()
                call.call_id = f"call_{digest[:16]}"
            else:
                call.request_index = self._next_request_index
                self._next_request_index += 1

    async def _forward(self, call: _Call, receive: Any, send: Any) -> None:
        assert self._client is not None and call.url is not None
        headers, call.changes = self._forward_headers(call.headers, call.body)
        try:
            # Built by hand and sent with send(): Client.build_request() would add the
            # client's default headers (User-Agent, Accept, Accept-Encoding).
            request = httpx.Request(call.method, call.url, headers=headers, content=call.body)
        except (httpx.InvalidURL, ValueError) as exc:
            await self._proxy_error(send, call, 400, f"the upstream URL is not valid: {exc}")
            return
        disconnected = asyncio.Event()
        watcher = asyncio.create_task(_watch_disconnect(receive, disconnected))
        sending = asyncio.create_task(self._client.send(request, stream=True))
        response: httpx.Response | None = None
        try:
            # Wait for the upstream's headers, but stop waiting as soon as the client leaves:
            # a local model can take minutes to answer, and nobody would read that answer.
            gone = asyncio.create_task(disconnected.wait())
            try:
                await asyncio.wait({sending, gone}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                gone.cancel()
            if not sending.done():
                call.end = "client_disconnected"
                return  # the finally block cancels the upstream request
            error = sending.exception()
            if isinstance(error, httpx.ConnectTimeout):
                await self._proxy_error(send, call, 504, f"connecting to the upstream timed out ({error})")
                return
            if isinstance(error, httpx.HTTPError):
                await self._proxy_error(
                    send, call, 502, f"the upstream could not be reached: {type(error).__name__}: {error}"
                )
                return
            if error is not None:
                raise error
            response = sending.result()
            call.first_byte = time.perf_counter()
            call.status = response.status_code
            call.response_headers = list(response.headers.raw)
            call.peer = _peer(response)
            if disconnected.is_set():
                call.end = "client_disconnected"  # the answer and the hang-up arrived together
                return
            out_headers, removed = _without_hop_by_hop(call.response_headers)
            call.changes += [f"response: dropped hop-by-hop header {name!r}" for name in removed]
            if 200 <= call.status < 300 and _is_responses_stream(call):
                call.assembler = ResponseStreamAssembler()
            await send({"type": "http.response.start", "status": call.status, "headers": out_headers})
            call.response_started = True
            call.end = await self._relay(response, send, call, disconnected)
            if call.end == "upstream_error":
                raise AbortConnection  # pass the cut on: the client must not see a clean end
            if call.end == "eof":
                await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            watcher.cancel()
            if not sending.done():
                sending.cancel()  # abandon the upstream request; httpx closes its connection
                await asyncio.wait({sending})
            if response is None and not sending.cancelled() and sending.exception() is None:
                response = sending.result()  # it answered after all: close that answer too
            if response is not None:
                with contextlib.suppress(Exception):
                    await response.aclose()

    async def _relay(
        self, response: httpx.Response, send: Any, call: _Call, disconnected: asyncio.Event
    ) -> str:
        """Pass each upstream chunk on as it arrives; return how the relay ended."""
        chunks = response.aiter_raw()
        gone = asyncio.create_task(disconnected.wait())
        read: asyncio.Task[bytes | None] | None = None
        try:
            while True:
                read = asyncio.create_task(_next_chunk(chunks))
                await asyncio.wait({read, gone}, return_when=asyncio.FIRST_COMPLETED)
                if not read.done():
                    return await self._client_left(call, chunks, read)
                try:
                    chunk = read.result()
                except httpx.HTTPError as exc:
                    call.upstream_error = f"{type(exc).__name__}: {exc}"
                    return "upstream_error"
                read = None
                if chunk is None:
                    return "eof"
                try:
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                except Exception as exc:  # uvicorn normally ignores sends after a disconnect
                    call.client_error = f"{type(exc).__name__}: {exc}"
                    disconnected.set()
                call.take(chunk)  # after the send, whether or not the client is still there
                if disconnected.is_set():
                    return await self._client_left(call, chunks, None)
        finally:
            gone.cancel()
            if read is not None and not read.done():
                read.cancel()
                read.add_done_callback(_retrieve)

    async def _client_left(
        self, call: _Call, chunks: AsyncIterator[bytes], read: asyncio.Task[bytes | None] | None
    ) -> str:
        """The client hung up. Close the upstream, unless the response had already ended."""
        if call.assembler is None or call.assembler.terminal_event is None:
            return "client_disconnected"
        # Codex stops reading at response.completed and may hang up at once. That is
        # normal: keep what the upstream still sends, briefly, so the record is whole.
        try:
            async with asyncio.timeout(DRAIN_AFTER_TERMINAL_S):
                while True:
                    chunk = await read if read is not None else await _next_chunk(chunks)
                    read = None
                    if chunk is None:
                        break
                    call.take(chunk)
        except (TimeoutError, httpx.HTTPError):
            pass
        return "client_disconnected"

    async def _proxy_error(self, send: Any, call: _Call, status: int, reason: str) -> None:
        """Answer with an error that the proxy itself generated, and record it as such."""
        body = json.dumps(
            {
                "error": {
                    "type": "codexprobe_proxy_error",
                    "message": reason,
                    "upstream_url": call.url,
                    "note": "generated by the CodexProbe proxy, not by the upstream",
                }
            }
        ).encode()
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
        if call.response_started:
            return  # too late for a status line: the client already has the upstream's
        call.proxy_status, call.proxy_headers, call.proxy_body, call.proxy_reason = status, headers, body, reason
        call.response_started = True
        with contextlib.suppress(Exception):
            await send({"type": "http.response.start", "status": status, "headers": headers})
            await send({"type": "http.response.body", "body": body, "more_body": False})

    def _forward_headers(self, headers: RawHeaders, body: bytes) -> tuple[RawHeaders, list[str]]:
        """The request headers to send upstream, and the list of changes made to them."""
        kept, removed = _without_hop_by_hop(headers)
        changes = [f"request: dropped hop-by-hop header {name!r}" for name in removed]
        mode = self.config.upstream.auth.mode
        forward: RawHeaders = []
        for name, value in kept:
            lower = name.decode("latin-1").lower()
            if lower == "host":
                changes.append("request: Host set to the upstream host")
            elif mode != "passthrough" and is_sensitive_header(lower):
                changes.append(f"request: dropped credential header {lower!r} (auth mode {mode})")
            else:
                forward.append((name, value))
        if mode == "env":
            forward.append((b"authorization", f"Bearer {self._auth_value}".encode("latin-1")))
            changes.append(
                f"request: authorization set from ${self.config.upstream.auth.env_var} (auth mode env)"
            )
        if body and not any(name.lower() == b"content-length" for name, _ in forward):
            changes.append("request: Content-Length set by the proxy (the body arrived chunked)")
        return forward, changes

    # --- records -------------------------------------------------------------

    def _finish(self, call: _Call) -> None:
        """Turn what was observed into a record and queue it. No disk I/O happens here."""
        finished = time.perf_counter()
        result = call.assembler.finish(_STREAM_END[call.end]) if call.assembler is not None else None
        request_body = encode_body(call.body)
        response_body = b"".join(call.chunks)
        timing = {
            "started_at": call.started_at,
            "first_byte_ms": _ms(call.arrival, call.first_byte),
            "first_chunk_ms": _ms(call.arrival, call.first_chunk),
            "total_ms": _ms(call.arrival, finished),
        }
        outcome = _outcome(call, result)
        errors = {"upstream": call.upstream_error, "client": call.client_error}
        response = _response_json(call, response_body)
        proxy_response = {"reason": call.proxy_reason} if call.proxy_reason is not None else None
        try:
            if self._writer is None:
                raise RecorderError("the recorder has no log writer")
            if not call.is_llm:
                self._writer.submit_request_log(
                    {
                        "kind": "request",
                        "index": call.request_index,
                        "method": call.method,
                        "path": call.path,
                        "query": call.query,
                        "url": call.url,
                        "headers": redact_headers(call.headers),
                        "body": request_body,
                        "response": response,
                        "proxy_response": proxy_response,
                        "outcome": outcome,
                        "timing": timing,
                        "errors": errors,
                        "changes": call.changes,
                    }
                )
                return
            self._writer.submit_call(
                {
                    "kind": "llm_call",
                    "seq": call.seq,
                    "call_id": call.call_id,
                    **call_identity(call.headers, request_body.get("json")),
                    "upstream": {
                        "name": self.config.upstream.name,
                        "url": call.url,
                        "peer": call.peer,
                        "changes": call.changes,
                    },
                    "request": {
                        "method": call.method,
                        "path": call.path,
                        "query": call.query,
                        "headers": redact_headers(call.headers),
                        "body": request_body,
                    },
                    "response": response,
                    "proxy_response": proxy_response,
                    "stream": _stream_json(result) if result is not None else None,
                    "timing": timing,
                    "outcome": outcome,
                    "errors": errors,
                    "derived": _derived(call, response_body, result),
                }
            )
        except (RecordError, RuntimeError) as exc:
            label = f"seq {call.seq}" if call.is_llm else f"request {call.request_index}"
            self._submit_errors.append(f"{label}: {type(exc).__name__}: {exc}")

    def _metadata(self, **extra: Any) -> dict[str, Any]:
        config = json.loads(json.dumps(dataclasses.asdict(self.config), default=str))
        return {
            "format": "codexprobe-run/1",
            "endpoint": self._endpoint,
            "started_at": self._started_at,
            "seed": self.config.seed,
            "config": config,  # the config holds no secrets: auth names an env variable at most
            "versions": _versions(),
            **extra,
        }


# --- helpers -----------------------------------------------------------------


async def _read_body(receive: Any) -> bytes | None:
    body = bytearray()
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        body += message.get("body", b"")
        if not message.get("more_body", False):
            return bytes(body)


async def _watch_disconnect(receive: Any, disconnected: asyncio.Event) -> None:
    """Set ``disconnected`` when the client goes away (call after the body was read)."""
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            disconnected.set()
            return


async def _next_chunk(chunks: AsyncIterator[bytes]) -> bytes | None:
    try:
        return await anext(chunks)
    except StopAsyncIteration:
        return None


def _retrieve(task: asyncio.Task[Any]) -> None:
    """Fetch an abandoned task's exception so asyncio does not log it as unretrieved."""
    if not task.cancelled():
        task.exception()


def _without_hop_by_hop(headers: RawHeaders) -> tuple[RawHeaders, list[str]]:
    """Drop hop-by-hop headers and the headers that ``Connection`` names; return kept and dropped."""
    named: set[str] = set()
    for name, value in headers:
        if name.lower() == b"connection":
            named.update(token.strip().lower() for token in value.decode("latin-1").split(",") if token.strip())
    kept: RawHeaders = []
    dropped: list[str] = []
    for name, value in headers:
        lower = name.decode("latin-1").lower()
        if lower in HOP_BY_HOP or lower in named:
            dropped.append(lower)
        else:
            kept.append((name, value))
    return kept, dropped


def _is_event_stream(headers: RawHeaders) -> bool:
    return any(
        name.lower() == b"content-type" and value.decode("latin-1").lower().startswith("text/event-stream")
        for name, value in headers
    )


def _is_responses_stream(call: _Call) -> bool:
    """Whether to parse a successful response as a Responses SSE stream.

    It is one when the upstream says ``Content-Type: text/event-stream``. For a
    ``POST /v1/responses`` whose JSON body asked for ``"stream": true``, it is one
    too when the upstream sends no Content-Type at all: OpenAI's ChatGPT Codex
    backend answers that way, and Codex reads the body as SSE regardless
    (VALIDATION.md, Phase 3). The body is never sniffed, an explicit other
    Content-Type is respected, and no header is added to what Codex receives.
    """
    if _is_event_stream(call.response_headers):
        return True
    has_content_type = any(name.lower() == b"content-type" for name, _ in call.response_headers)
    return call.is_llm and not has_content_type and _asks_for_stream(call.body)


def _asks_for_stream(body: bytes) -> bool:
    """True if the request body is a JSON object whose top-level ``stream`` is the boolean true."""
    request = encode_body(body).get("json")  # standard JSON in UTF-8, as the record's json view
    return isinstance(request, dict) and request.get("stream") is True


def _peer(response: httpx.Response) -> str | None:
    stream = response.extensions.get("network_stream")
    try:
        address = stream.get_extra_info("server_addr") if stream is not None else None
    except Exception:
        return None
    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return None


def _response_json(call: _Call, upstream_body: bytes) -> dict[str, Any]:
    """The response the client received: generated by the proxy, the upstream's, or none at all."""
    if call.proxy_status is not None:
        return {
            "source": "proxy",
            "status": call.proxy_status,
            "headers": redact_headers(call.proxy_headers or []),
            "body": encode_body(call.proxy_body or b""),
        }
    if call.status is not None:
        return {
            "source": "upstream",
            "status": call.status,
            "headers": redact_headers(call.response_headers or []),
            "body": encode_body(upstream_body),
        }
    return {"source": None, "status": None, "headers": None, "body": None}


def _outcome(call: _Call, result: StreamResult | None) -> str:
    if call.proxy_status is not None:  # the proxy answered by itself
        return "recorder_stopped" if call.end == "recorder_stopped" else "proxy_error"
    if call.status is None:  # no response at all
        return call.end if call.end in ("client_disconnected", "recorder_stopped") else "proxy_error"
    if call.status >= 400:
        return "upstream_http_error"
    if result is not None:  # a terminal event decides, whatever happened afterwards
        return result.outcome
    if call.end != "eof" and _received_declared_length(call):
        return "completed"  # every byte the upstream announced arrived before the interruption
    return _PLAIN_OUTCOME[call.end]


def _received_declared_length(call: _Call) -> bool:
    """True if the upstream sent Content-Length and exactly that many body bytes arrived."""
    for name, value in call.response_headers or []:
        if name.lower() == b"content-length":
            try:
                return int(value) == sum(len(chunk) for chunk in call.chunks)
            except ValueError:
                return False
    return False


def _stream_json(result: StreamResult) -> dict[str, Any]:
    return {
        "events": [_event_json(event) for event in result.events],
        "reconstruction": {
            "outcome": result.outcome,
            "terminal_event": result.terminal_event,
            "response_id": result.response_id,
            "usage": result.usage,
            "items": result.items,
            "partial_items": [dataclasses.asdict(item) for item in result.partial_items],
            "incomplete_reason": result.incomplete_reason,
            "error": result.error,
            "parse_errors": result.parse_errors,
            "comment_count": result.comment_count,
            "ended_mid_event": result.ended_mid_event,
        },
    }


def _event_json(event: ParsedEvent) -> dict[str, Any]:
    entry: dict[str, Any] = {"index": event.index, "event": event.event, "id": event.id, "type": event.type}
    if event.json is not None:
        entry["json"] = event.json
    else:
        entry["data"] = event.data  # not JSON: keep the text (it is also in the raw body)
    if event.error is not None:
        entry["error"] = event.error
    return entry


def _derived(call: _Call, response_body: bytes, result: StreamResult | None) -> dict[str, Any]:
    items = result.items if result is not None else []
    return {
        "request_bytes": len(call.body),
        "response_bytes": len(response_body),
        "response_chunks": len(call.chunks),
        "event_count": len(result.events) if result is not None else None,
        "item_types": [item.get("type") for item in items],
        "tool_calls": [
            {"type": item.get("type"), "name": item.get("name"), "call_id": item.get("call_id")}
            for item in items
            if item.get("type") in ("function_call", "custom_tool_call")
        ],
    }


def _ms(start: float, moment: float | None) -> float | None:
    return None if moment is None else round((moment - start) * 1000, 3)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _versions() -> dict[str, str]:
    try:
        probe = importlib.metadata.version("codex-probe")
    except importlib.metadata.PackageNotFoundError:
        probe = "unknown"
    return {
        "python": platform.python_version(),
        "codex_probe": probe,
        "httpx": httpx.__version__,
        "uvicorn": uvicorn.__version__,
        "platform": platform.platform(),
    }
