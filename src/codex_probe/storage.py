"""JSONL storage for recorded calls.

This module provides:

* lossless body encoding (:func:`encode_body` / :func:`decode_body`);
* redaction of credential headers (:func:`redact_headers`);
* the identifiers that group Codex calls into sessions (:func:`call_identity`);
* the schema of one call record (:func:`validate_call_record`);
* a background writer for one run directory (:class:`RunWriter`);
* :func:`load_session`, which reads the calls back.

Layout of a run directory::

    <run_dir>/
        run.json                    run metadata (rewritten in place)
        session-<sha256>.jsonl      LLM calls of one Codex session, one record per line
        no-session.jsonl            LLM calls that carried no session identifier
        other-requests.jsonl        non-LLM requests such as GET /v1/models

File names are derived from a SHA-256 digest, so an untrusted session value
can never pick a path.

Deliberately not implemented: write-ahead logs, manifests, per-line hashes and
crash recovery. A crash can lose calls that were still in flight, and a line
cut off mid-write makes :func:`load_session` fail loudly instead of guessing.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import queue
import threading
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, BinaryIO

REDACTED = "[redacted]"
RUN_METADATA_FILE = "run.json"
NO_SESSION_FILE = "no-session.jsonl"
OTHER_REQUESTS_FILE = "other-requests.jsonl"
SESSION_FILE_PREFIX = "session-"

OUTCOMES = frozenset(
    {
        "completed",
        "incomplete",
        "failed",
        "stream_cut",
        "client_disconnected",
        "recorder_stopped",
        "upstream_http_error",
        "proxy_error",
    }
)

# Header names whose values are credentials. Matching ignores case.
_SENSITIVE_NAMES = frozenset(
    {"authorization", "proxy-authorization", "cookie", "set-cookie", "chatgpt-account-id"}
)
_SENSITIVE_PARTS = ("authorization", "cookie", "api-key", "api_key", "apikey", "secret", "password")

_RECORD_FIELDS = (
    "kind",
    "seq",
    "call_id",
    "session_key",
    "thread_id",
    "turn_id",
    "upstream",
    "request",
    "response",
    "stream",
    "timing",
    "outcome",
    "derived",
)

Header = tuple[str | bytes, str | bytes]


class RecordError(ValueError):
    """A record does not match the call-record schema; the message names the field."""


class SessionLoadError(ValueError):
    """A run directory or JSONL file cannot be loaded; the message names the file and line."""


class LogWriteError(RuntimeError):
    """Writing the logs failed. Every call recorded in memory is still available in ``calls``."""

    def __init__(self, message: str, calls: list[dict[str, Any]], errors: list[str]) -> None:
        super().__init__(message)
        self.calls = calls
        self.errors = errors


# --- bodies and headers ------------------------------------------------------


def encode_body(data: bytes) -> dict[str, Any]:
    """Represent bytes losslessly.

    Valid UTF-8 becomes ``{"text": ...}`` and anything else ``{"base64": ...}``.
    Text that is also standard JSON gets an extra ``"json"`` view.
    :func:`decode_body` turns either form back into the exact bytes.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"base64": base64.b64encode(data).decode("ascii")}
    body: dict[str, Any] = {"text": text}
    try:
        body["json"] = json.loads(text, parse_constant=_reject_constant)
    except ValueError:
        pass  # not JSON: the text alone is the record
    return body


def decode_body(body: Mapping[str, Any]) -> bytes:
    """Return the exact bytes that an :func:`encode_body` result stands for."""
    if "text" in body:
        return str(body["text"]).encode("utf-8")
    return base64.b64decode(body["base64"], validate=True)


def is_sensitive_header(name: str) -> bool:
    """True if a header with this name carries a credential and must not be logged."""
    lowered = name.lower()
    return (
        lowered in _SENSITIVE_NAMES
        or any(part in lowered for part in _SENSITIVE_PARTS)
        or lowered.endswith("token")
    )


def redact_headers(headers: Iterable[Header]) -> list[list[str]]:
    """Copy headers in order, duplicates included, replacing credential values with ``[redacted]``.

    Bytes are decoded as latin-1, which maps every byte to one character, so
    nothing is lost.
    """
    redacted: list[list[str]] = []
    for name, value in headers:
        name_text = _latin1(name)
        redacted.append([name_text, REDACTED if is_sensitive_header(name_text) else _latin1(value)])
    return redacted


def call_identity(headers: Iterable[Header], body_json: Any) -> dict[str, str | None]:
    """Return the session key, thread id and turn id of a Codex request.

    * ``session_key``: the ``session-id`` header, else ``client_metadata.session_id``.
      It stays the same for a whole ``codex exec`` run, sub-agents included.
    * ``thread_id``: the ``thread-id`` header, else ``client_metadata.thread_id``.
    * ``turn_id``: ``client_metadata.turn_id``, else the ``turn_id`` in the
      ``x-codex-turn-metadata`` header.

    ``body_json`` is the parsed request body, or None if it was not JSON.
    """
    first: dict[str, str] = {}
    for name, value in headers:
        first.setdefault(_latin1(name).lower(), _latin1(value))
    metadata = body_json.get("client_metadata") if isinstance(body_json, dict) else None
    if not isinstance(metadata, dict):
        metadata = {}
    try:
        turn_metadata = json.loads(first.get("x-codex-turn-metadata", ""))
    except ValueError:
        turn_metadata = None
    if not isinstance(turn_metadata, dict):
        turn_metadata = {}
    return {
        "session_key": _first_text(first.get("session-id"), metadata.get("session_id")),
        "thread_id": _first_text(first.get("thread-id"), metadata.get("thread_id")),
        "turn_id": _first_text(metadata.get("turn_id"), turn_metadata.get("turn_id")),
    }


def session_file_name(session_key: str | None) -> str:
    """The JSONL file that holds a session's calls. The name is a digest, never the raw key."""
    if session_key is None:
        return NO_SESSION_FILE
    digest = hashlib.sha256(session_key.encode("utf-8", "surrogatepass")).hexdigest()
    return f"{SESSION_FILE_PREFIX}{digest}.jsonl"


# --- the call-record schema --------------------------------------------------


def validate_call_record(record: Any) -> None:
    """Check that a call record has every required field, with the right type.

    The record has separate layers:

    * the raw HTTP exchange: ``request`` as the client sent it, and ``response`` as
      the client received it, with ``response.source`` saying who produced it;
    * the output rebuilt from the stream: ``stream``, holding the parsed events
      and the reconstruction;
    * statistics we computed ourselves: ``derived``.

    Extra fields are allowed. Credential headers must already be redacted.

    Raises:
        RecordError: naming the first problem found.
    """
    rec = _object(record, "record")
    for name in _RECORD_FIELDS:
        _field(rec, name, "record")
    if rec["kind"] != "llm_call":
        raise RecordError(f"record.kind must be 'llm_call', got {rec['kind']!r}")
    _non_negative_int(rec["seq"], "record.seq")
    _string(rec["call_id"], "record.call_id", allow_empty=False)
    for name in ("session_key", "thread_id", "turn_id"):
        _optional_string(rec[name], f"record.{name}")

    upstream = _object(rec["upstream"], "record.upstream")
    _string(_field(upstream, "name", "record.upstream"), "record.upstream.name", allow_empty=False)
    _string(_field(upstream, "url", "record.upstream"), "record.upstream.url", allow_empty=False)
    _optional_string(_field(upstream, "peer", "record.upstream"), "record.upstream.peer")

    request = _object(rec["request"], "record.request")
    _string(_field(request, "method", "record.request"), "record.request.method", allow_empty=False)
    _string(_field(request, "path", "record.request"), "record.request.path", allow_empty=False)
    _headers(_field(request, "headers", "record.request"), "record.request.headers")
    _body(_field(request, "body", "record.request"), "record.request.body")

    # The response the client received: the upstream's ("upstream"), or one the proxy
    # generated itself ("proxy"). All fields are null only when no response was
    # produced at all, e.g. the client left before the upstream answered.
    response = _object(rec["response"], "record.response")
    source = _field(response, "source", "record.response")
    status = _field(response, "status", "record.response")
    headers = _field(response, "headers", "record.response")
    body = _field(response, "body", "record.response")
    if source is None:
        if (status, headers, body) != (None, None, None):
            raise RecordError("record.response: status, headers and body must be null when source is null")
    elif source in ("upstream", "proxy"):
        _non_negative_int(status, "record.response.status")
        _headers(headers, "record.response.headers")
        _body(body, "record.response.body")
    else:
        raise RecordError(f"record.response.source must be 'upstream', 'proxy' or null; got {source!r}")

    if rec["stream"] is not None:  # null for a response that was not an SSE stream
        stream = _object(rec["stream"], "record.stream")
        if not isinstance(_field(stream, "events", "record.stream"), list):
            raise RecordError("record.stream.events must be a list")
        _object(_field(stream, "reconstruction", "record.stream"), "record.stream.reconstruction")

    timing = _object(rec["timing"], "record.timing")
    _string(_field(timing, "started_at", "record.timing"), "record.timing.started_at", allow_empty=False)
    for name in ("first_byte_ms", "total_ms"):
        _optional_duration(_field(timing, name, "record.timing"), f"record.timing.{name}")

    if rec["outcome"] not in OUTCOMES:
        raise RecordError(
            f"record.outcome must be one of {', '.join(sorted(OUTCOMES))}; got {rec['outcome']!r}"
        )
    _object(rec["derived"], "record.derived")


# --- the background writer ---------------------------------------------------


class RunWriter:
    """Write one run's records to JSONL files on a background thread.

    ``submit_*`` and :meth:`write_metadata` only validate and queue, and never
    touch the disk, so they are safe to call from the forwarding path. The
    writer thread creates the directory, appends one line per record, and
    flushes after every line. :meth:`stop` waits until the queue is drained.
    """

    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        self.run_dir = Path(run_dir)
        self._queue: queue.SimpleQueue[tuple[str, Any] | None] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._calls: list[dict[str, Any]] = []
        self._seqs: set[int] = set()
        self._errors: list[str] = []
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run, name="codexprobe-log-writer", daemon=True
        )
        self._thread.start()

    def submit_call(self, record: dict[str, Any]) -> None:
        """Validate a finished LLM call record and queue it for writing.

        Raises:
            RecordError: the record is invalid, or its ``seq`` was already submitted.
            RuntimeError: the writer has been stopped.
        """
        validate_call_record(record)
        with self._lock:
            self._check_running()
            if record["seq"] in self._seqs:
                raise RecordError(f"record.seq {record['seq']} was already submitted")
            self._seqs.add(record["seq"])
            self._calls.append(record)
            self._queue.put(("call", record))

    def submit_request_log(self, record: dict[str, Any]) -> None:
        """Queue a record of a non-LLM request (``kind: "request"``).

        These records go to ``other-requests.jsonl`` and never appear in the calls.
        """
        if not isinstance(record, dict) or record.get("kind") != "request":
            raise RecordError("a request log record must be an object with kind 'request'")
        if "headers" in record:
            _headers(record["headers"], "request_log.headers")
        with self._lock:
            self._check_running()
            self._queue.put(("request", record))

    def write_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Queue a rewrite of ``run.json``. The last version queued wins."""
        with self._lock:
            self._check_running()
            self._queue.put(("metadata", dict(metadata)))

    def stop(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """Wait until everything queued is written, then return the calls ordered by ``seq``.

        Raises:
            LogWriteError: a write failed, or the writer did not finish within
                ``timeout`` seconds. The exception carries the calls.
        """
        with self._lock:
            if not self._stopped:
                self._stopped = True
                self._queue.put(None)
        self._thread.join(timeout)
        calls = sorted(self._calls, key=lambda record: record["seq"])
        if self._thread.is_alive():
            raise LogWriteError(
                f"the log writer did not finish within {timeout} s", calls, list(self._errors)
            )
        if self._errors:
            raise LogWriteError(
                f"{len(self._errors)} log write(s) failed; first: {self._errors[0]}",
                calls,
                list(self._errors),
            )
        return calls

    def _check_running(self) -> None:
        if self._stopped:
            raise RuntimeError("the log writer has been stopped")

    def _run(self) -> None:
        handles: dict[str, BinaryIO] = {}
        directory_ready = False
        try:
            while (item := self._queue.get()) is not None:
                kind, payload = item
                try:
                    if not directory_ready:
                        self.run_dir.mkdir(parents=True, exist_ok=True)
                        directory_ready = True
                    self._write(kind, payload, handles)
                except Exception as exc:  # reported by stop(); never swallowed
                    self._errors.append(f"{kind} {_describe(payload)}: {type(exc).__name__}: {exc}")
        finally:
            for name, handle in handles.items():
                try:
                    handle.close()
                except OSError as exc:
                    self._errors.append(f"closing {name}: {type(exc).__name__}: {exc}")

    def _write(self, kind: str, payload: Any, handles: dict[str, BinaryIO]) -> None:
        if kind == "metadata":
            temporary = self.run_dir / (RUN_METADATA_FILE + ".tmp")
            temporary.write_bytes(_dump(payload, indent=2) + b"\n")
            os.replace(temporary, self.run_dir / RUN_METADATA_FILE)
            return
        name = OTHER_REQUESTS_FILE if kind == "request" else session_file_name(payload["session_key"])
        line = _dump(payload) + b"\n"  # serialize first, so a failure writes nothing
        handle = handles.get(name)
        if handle is None:
            handle = handles[name] = open(self.run_dir / name, "ab")
        handle.write(line)
        handle.flush()


# --- reading a run back ------------------------------------------------------


def load_session(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read the LLM calls of a run directory, or of one session JSONL file, ordered by ``seq``.

    For a run directory, every ``session-*.jsonl`` and ``no-session.jsonl``
    file is read; ``run.json`` and ``other-requests.jsonl`` are not calls and
    are skipped. Each line must be a complete JSON object that passes
    :func:`validate_call_record`.

    Raises:
        SessionLoadError: naming the file and line of the first problem: an
            empty file, a cut-off or blank line, invalid JSON, a schema error,
            a record in the wrong session file, or a duplicate ``seq``.
    """
    target = Path(path)
    if target.is_dir():
        files = sorted(p for p in target.iterdir() if p.is_file() and _is_call_file(p.name))
        check_file_names = True
    elif target.is_file():
        files = [target]
        check_file_names = False
    else:
        raise SessionLoadError(f"{target}: no such run directory or JSONL file")
    records: list[dict[str, Any]] = []
    seen: dict[int, str] = {}
    for file in files:
        for line_number, record in _read_call_file(file, check_file_names):
            where = f"{file}:{line_number}"
            if record["seq"] in seen:
                raise SessionLoadError(f"{where}: seq {record['seq']} also appears at {seen[record['seq']]}")
            seen[record["seq"]] = where
            records.append(record)
    records.sort(key=lambda record: record["seq"])
    return records


def _read_call_file(file: Path, check_file_name: bool) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        data = file.read_bytes()
    except OSError as exc:
        raise SessionLoadError(f"{file}: cannot read: {exc}") from None
    if not data:
        raise SessionLoadError(f"{file}: file is empty")
    lines = data.split(b"\n")
    if lines[-1]:
        raise SessionLoadError(
            f"{file}:{len(lines)}: last line has no newline; the file was cut off mid-write"
        )
    for line_number, raw in enumerate(lines[:-1], start=1):
        where = f"{file}:{line_number}"
        if not raw.strip():
            raise SessionLoadError(f"{where}: blank line")
        try:
            record = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError:
            raise SessionLoadError(f"{where}: not valid UTF-8") from None
        except ValueError as exc:
            raise SessionLoadError(f"{where}: not valid JSON ({exc})") from None
        if not isinstance(record, dict):
            raise SessionLoadError(f"{where}: line is not a JSON object")
        try:
            validate_call_record(record)
        except RecordError as exc:
            raise SessionLoadError(f"{where}: {exc}") from None
        if check_file_name and file.name != session_file_name(record["session_key"]):
            raise SessionLoadError(
                f"{where}: this record belongs in {session_file_name(record['session_key'])}"
            )
        yield line_number, record


def _is_call_file(name: str) -> bool:
    return name == NO_SESSION_FILE or (
        name.startswith(SESSION_FILE_PREFIX) and name.endswith(".jsonl")
    )


# --- helpers -----------------------------------------------------------------


def _dump(value: Any, indent: int | None = None) -> bytes:
    """Serialize to UTF-8 JSON. Readable text where possible; \\u escapes if a lone surrogate needs them."""
    separators = None if indent else (",", ":")
    try:
        return json.dumps(value, ensure_ascii=False, indent=indent, separators=separators).encode(
            "utf-8"
        )
    except UnicodeEncodeError:
        return json.dumps(value, ensure_ascii=True, indent=indent, separators=separators).encode(
            "ascii"
        )


def _describe(payload: Any) -> str:
    if isinstance(payload, dict) and "seq" in payload:
        return f"seq {payload['seq']}"
    return ""


def _reject_constant(name: str) -> None:
    raise ValueError(f"{name} is not standard JSON")


def _latin1(value: str | bytes) -> str:
    return value.decode("latin-1") if isinstance(value, bytes) else value


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _field(obj: Mapping[str, Any], name: str, path: str) -> Any:
    if name not in obj:
        raise RecordError(f"{path}.{name}: required field is missing")
    return obj[name]


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RecordError(f"{path} must be an object")
    return value


def _string(value: Any, path: str, *, allow_empty: bool = True) -> None:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise RecordError(f"{path} must be a {'' if allow_empty else 'non-empty '}string")


def _optional_string(value: Any, path: str) -> None:
    if value is not None and not isinstance(value, str):
        raise RecordError(f"{path} must be a string or null")


def _non_negative_int(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecordError(f"{path} must be a non-negative integer")


def _optional_duration(value: Any, path: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise RecordError(f"{path} must be a non-negative number or null")


def _headers(value: Any, path: str) -> None:
    if not isinstance(value, list):
        raise RecordError(f"{path} must be a list of [name, value] pairs")
    for index, pair in enumerate(value):
        if not (
            isinstance(pair, list) and len(pair) == 2 and all(isinstance(part, str) for part in pair)
        ):
            raise RecordError(f"{path}[{index}] must be a [name, value] pair of strings")
        if is_sensitive_header(pair[0]) and pair[1] != REDACTED:
            # The value is not echoed: it is a credential.
            raise RecordError(f"{path}[{index}]: credential header {pair[0]!r} must be redacted")


def _body(value: Any, path: str) -> None:
    body = _object(value, path)
    unknown = sorted(set(body) - {"text", "base64", "json"})
    if unknown:
        raise RecordError(f"{path}: unknown field(s) {', '.join(unknown)}")
    if ("text" in body) == ("base64" in body):
        raise RecordError(f"{path} must have exactly one of 'text' or 'base64'")
    if "text" in body:
        _string(body["text"], f"{path}.text")
        return
    if "json" in body:
        raise RecordError(f"{path}: a base64 body cannot have a 'json' view")
    _string(body["base64"], f"{path}.base64")
    try:
        base64.b64decode(body["base64"], validate=True)
    except (binascii.Error, ValueError):
        raise RecordError(f"{path}.base64 is not valid base64") from None
