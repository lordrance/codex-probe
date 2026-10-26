"""Incremental SSE parsing and reassembly of OpenAI Responses API streams.

The proxy relays each upstream chunk to Codex first and then feeds a copy to
:class:`ResponseStreamAssembler`, so nothing in this module can change or stop
the relay. A malformed event is recorded as a parse error, and parsing goes on.

The module has two layers:

* :class:`SSEParser` turns raw bytes into Server-Sent Events, following the
  WHATWG event-stream rules. Lines may end in CRLF, LF or CR, comment lines are
  skipped, and multi-line ``data:`` fields are joined. An event is dispatched
  only at a blank line, so an event cut off at the end of the stream is never
  dispatched.
* :class:`ResponseStreamAssembler` reads the JSON ``type`` of each event and
  rebuilds the model output. Like Codex 0.164.0, it ignores the SSE ``event:``
  name for dispatch. The complete items in ``response.output_item.done`` are
  authoritative. Deltas are used only to show a partial result when the stream
  stops before any terminal event.

The parsed view is derived data. The raw bytes, kept separately by the proxy,
remain the record of what was actually sent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

Outcome = Literal[
    "completed",            # a valid response.completed was parsed
    "incomplete",           # response.incomplete
    "failed",               # response.failed or an error event
    "stream_cut",           # upstream closed before any terminal event
    "client_disconnected",  # the client went away before any terminal event
    "recorder_stopped",     # the recorder stopped before any terminal event
]
StreamEnd = Literal["eof", "client_disconnected", "recorder_stopped"]

TERMINAL_EVENT_TYPES: tuple[str, ...] = (
    "response.completed",
    "response.incomplete",
    "response.failed",
    "error",
)

_END_OUTCOME: dict[str, Outcome] = {
    "eof": "stream_cut",
    "client_disconnected": "client_disconnected",
    "recorder_stopped": "recorder_stopped",
}
_LINE_END = re.compile(rb"\r\n|\r|\n")
_USAGE_TOTALS = ("input_tokens", "output_tokens", "total_tokens")

# Identifies an output item: ("id", item id), ("index", output_index) or ("call", call_id).
_ItemKey = tuple[str, Any]


@dataclass(frozen=True)
class SSEEvent:
    """One dispatched Server-Sent Event."""

    index: int  # position among the dispatched events, starting at 0
    event: str | None  # value of the ``event:`` field, if the event had one
    data: str  # the ``data:`` lines joined with "\n"
    id: str | None  # the last event ID seen so far


class SSEParser:
    """Incremental SSE parser. Feed it bytes in chunks of any size."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._at_stream_start = True
        self._data_lines: list[str] = []
        self._event_name: str | None = None
        self._last_id: str | None = None
        self._has_fields = False  # a field has been seen since the last blank line
        self._dispatched = 0
        self._closed = False
        self.comment_count = 0
        """Comment lines seen so far, e.g. ``: ping`` heartbeats."""

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        """Parse a chunk and return the events it completed."""
        if self._closed:
            raise RuntimeError("SSEParser.feed() called after close()")
        if not chunk:
            return []
        scan_from = len(self._buffer)
        if scan_from and self._buffer[-1] == 0x0D:
            scan_from -= 1  # re-examine the held-back CR: it may start a CRLF
        self._buffer += chunk
        return self._drain(scan_from, at_eof=False)

    def close(self) -> list[SSEEvent]:
        """Mark the end of the stream and return any event a trailing CR completed.

        An event with no blank line after it is not dispatched; see
        :attr:`ended_mid_event`.
        """
        if self._closed:
            return []
        self._closed = True
        return self._drain(0, at_eof=True)

    @property
    def ended_mid_event(self) -> bool:
        """True if the bytes after the last blank line held an unfinished line or event."""
        return bool(self._buffer) or self._has_fields

    def _drain(self, scan_from: int, *, at_eof: bool) -> list[SSEEvent]:
        events: list[SSEEvent] = []
        buffer = self._buffer
        line_start = 0
        for match in _LINE_END.finditer(buffer, scan_from):
            if not at_eof and match.group() == b"\r" and match.end() == len(buffer):
                break  # a CR at the end of a chunk may be the first half of CRLF
            event = self._process_line(bytes(buffer[line_start : match.start()]))
            line_start = match.end()
            if event is not None:
                events.append(event)
        del buffer[:line_start]
        return events

    def _process_line(self, raw: bytes) -> SSEEvent | None:
        # Lines are split on bytes and decoded only when complete, so a UTF-8
        # character split across chunks is decoded correctly.
        line = raw.decode("utf-8", errors="replace")
        if self._at_stream_start:
            self._at_stream_start = False
            line = line.removeprefix("﻿")
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            self.comment_count += 1
            return None
        self._has_fields = True
        name, colon, value = line.partition(":")
        if colon and value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._data_lines.append(value)
        elif name == "event":
            self._event_name = value
        elif name == "id" and "\x00" not in value:
            self._last_id = value
        # "retry" and unknown field names carry nothing we need; the spec says to ignore them.
        return None

    def _dispatch(self) -> SSEEvent | None:
        event = None
        if self._data_lines:  # the spec does not dispatch an event that had no data field
            event = SSEEvent(
                index=self._dispatched,
                event=self._event_name,
                data="\n".join(self._data_lines),
                id=self._last_id,
            )
            self._dispatched += 1
        self._data_lines = []
        self._event_name = None
        self._has_fields = False
        return event


@dataclass(frozen=True)
class ParsedEvent:
    """An SSE event together with its JSON payload, as seen by the recorder."""

    index: int
    event: str | None
    data: str
    id: str | None
    json: Any  # the parsed ``data``; None if it was not valid JSON
    type: str | None  # the JSON ``type`` field, which is what Codex dispatches on
    error: str | None = None  # why the event could not be used, if it could not


@dataclass(frozen=True)
class PartialItem:
    """Output rebuilt from deltas for an item whose ``response.output_item.done`` never arrived.

    Produced only when the stream stopped before any terminal event, so it is
    always an incomplete view of what the model was generating.
    """

    delta_type: str  # e.g. "response.output_text.delta"
    item_id: str | None
    output_index: int | None
    call_id: str | None
    added_item: dict[str, Any] | None  # the item from response.output_item.added, if any
    text: str  # the deltas, concatenated in arrival order


@dataclass(frozen=True)
class StreamResult:
    """What the recorder reconstructed from one Responses SSE stream."""

    outcome: Outcome
    terminal_event: str | None  # type of the event that ended the response, if one did
    response_id: str | None
    usage: dict[str, Any] | None
    items: list[dict[str, Any]]  # complete items from response.output_item.done, in order
    partial_items: list[PartialItem]  # empty unless the stream stopped before a terminal event
    incomplete_reason: str | None  # response.incomplete_details.reason
    error: dict[str, Any] | None  # the error object of response.failed or an error event
    events: list[ParsedEvent]
    parse_errors: list[str]
    comment_count: int
    ended_mid_event: bool

    @property
    def completed(self) -> bool:
        """True only if a complete, valid ``response.completed`` event was parsed."""
        return self.outcome == "completed"


@dataclass
class _DeltaBuffer:
    delta_type: str
    item_id: str | None
    output_index: int | None
    call_id: str | None
    keys: set[_ItemKey] = field(default_factory=set)
    parts: list[str] = field(default_factory=list)


class ResponseStreamAssembler:
    """Rebuild a Responses API stream. Feed raw SSE bytes, then call :meth:`finish` once."""

    def __init__(self) -> None:
        self._parser = SSEParser()
        self._events: list[ParsedEvent] = []
        self._parse_errors: list[str] = []
        self._items: list[dict[str, Any]] = []
        self._done_keys: set[_ItemKey] = set()
        self._added_items: dict[_ItemKey, dict[str, Any]] = {}
        self._deltas: dict[tuple[_ItemKey, str], _DeltaBuffer] = {}
        self._terminal_event: str | None = None
        self._outcome: Outcome | None = None
        self._response_id: str | None = None
        self._usage: dict[str, Any] | None = None
        self._incomplete_reason: str | None = None
        self._error: dict[str, Any] | None = None
        self._result: StreamResult | None = None

    @property
    def terminal_event(self) -> str | None:
        """The type of the terminal event parsed so far, or None."""
        return self._terminal_event

    def feed(self, chunk: bytes) -> None:
        """Parse a chunk of raw SSE bytes. Chunks may be split anywhere."""
        if self._result is not None:
            raise RuntimeError("ResponseStreamAssembler.feed() called after finish()")
        for event in self._parser.feed(chunk):
            self._handle(event)

    def finish(self, end: StreamEnd = "eof") -> StreamResult:
        """End the stream and return the reconstruction.

        ``end`` says why the stream stopped: ``"eof"`` (the upstream closed it),
        ``"client_disconnected"`` or ``"recorder_stopped"``. It only matters
        if no terminal event was parsed. After a terminal event, the outcome
        comes from that event: a client that disconnects right after
        ``response.completed`` is normal, because Codex stops reading there.
        Later calls return the same result.
        """
        if end not in _END_OUTCOME:
            raise ValueError(f"end must be one of {', '.join(_END_OUTCOME)}, got {end!r}")
        if self._result is not None:
            return self._result
        for event in self._parser.close():
            self._handle(event)
        parse_errors = list(self._parse_errors)
        if self._parser.ended_mid_event:
            parse_errors.append(
                "the stream ended in the middle of an event; that event was not dispatched"
            )
        self._result = StreamResult(
            outcome=self._outcome if self._outcome is not None else _END_OUTCOME[end],
            terminal_event=self._terminal_event,
            response_id=self._response_id,
            usage=self._usage,
            items=list(self._items),
            partial_items=self._partial_items() if self._terminal_event is None else [],
            incomplete_reason=self._incomplete_reason,
            error=self._error,
            events=list(self._events),
            parse_errors=parse_errors,
            comment_count=self._parser.comment_count,
            ended_mid_event=self._parser.ended_mid_event,
        )
        return self._result

    def _handle(self, sse: SSEEvent) -> None:
        try:
            payload = json.loads(sse.data)
        except ValueError as exc:
            self._record(sse, None, None, f"data is not valid JSON ({exc})")
            return
        if not isinstance(payload, dict):
            self._record(sse, payload, None, "data is not a JSON object")
            return
        event_type = payload.get("type")
        if not isinstance(event_type, str):
            self._record(sse, payload, None, "the JSON has no string 'type' field")
            return
        if self._terminal_event is not None:
            problem: str | None = (
                f"arrived after the terminal event {self._terminal_event}; ignored"
            )
        else:
            problem = self._apply(event_type, payload)
        self._record(sse, payload, event_type, problem)

    def _record(
        self, sse: SSEEvent, payload: Any, event_type: str | None, problem: str | None
    ) -> None:
        self._events.append(
            ParsedEvent(
                index=sse.index,
                event=sse.event,
                data=sse.data,
                id=sse.id,
                json=payload,
                type=event_type,
                error=problem,
            )
        )
        if problem is not None:
            self._parse_errors.append(f"event {sse.index} ({event_type or 'no type'}): {problem}")

    def _apply(self, event_type: str, payload: dict[str, Any]) -> str | None:
        """Update the reconstruction; return why the event could not be used, if it could not."""
        if event_type in TERMINAL_EVENT_TYPES:
            return self._apply_terminal(event_type, payload)
        if event_type in ("response.created", "response.in_progress"):
            response = payload.get("response")
            if isinstance(response, dict) and isinstance(response.get("id"), str):
                self._response_id = self._response_id or response["id"]
            return None
        if event_type == "response.output_item.added":
            item = payload.get("item")
            if not isinstance(item, dict):
                return "no 'item' object"
            for key in _item_keys(payload, item):
                self._added_items.setdefault(key, item)
            return None
        if event_type == "response.output_item.done":
            item = payload.get("item")
            if not isinstance(item, dict):
                return "no 'item' object"
            self._items.append(item)
            self._done_keys |= _item_keys(payload, item)
            return None
        if event_type.endswith(".delta"):
            delta = payload.get("delta")
            if not isinstance(delta, str):
                return "no string 'delta'"
            self._add_delta(event_type, payload, delta)
            return None
        return None  # other event types carry no output; they stay in the event list

    def _apply_terminal(self, event_type: str, payload: dict[str, Any]) -> str | None:
        response = payload.get("response")
        if event_type == "response.completed":
            problem = _completed_problem(response)
            if problem is not None:
                # Not a completion. Codex keeps reading too, so a later valid one still counts.
                return problem
            self._end_response(event_type, "completed", response)
            return None
        if event_type == "response.incomplete":
            self._end_response(event_type, "incomplete", response)
            details = response.get("incomplete_details") if isinstance(response, dict) else None
            reason = details.get("reason") if isinstance(details, dict) else None
            self._incomplete_reason = reason if isinstance(reason, str) else None
        elif event_type == "response.failed":
            self._end_response(event_type, "failed", response)
            error = response.get("error") if isinstance(response, dict) else None
            self._error = error if isinstance(error, dict) else None
        else:  # "error": the details are nested under "error" or sit at the top level
            self._end_response(event_type, "failed", response)
            nested = payload.get("error")
            if isinstance(nested, dict):
                self._error = nested
            else:
                self._error = {k: payload[k] for k in ("code", "message", "param") if k in payload}
        return None

    def _end_response(self, event_type: str, outcome: Outcome, response: Any) -> None:
        self._terminal_event = event_type
        self._outcome = outcome
        if isinstance(response, dict):
            if isinstance(response.get("id"), str):
                self._response_id = response["id"]
            if isinstance(response.get("usage"), dict):
                self._usage = response["usage"]

    def _add_delta(self, event_type: str, payload: dict[str, Any], delta: str) -> None:
        keys = _item_keys(payload)
        slot = (_primary_key(keys), event_type)
        buffer = self._deltas.get(slot)
        if buffer is None:
            buffer = _DeltaBuffer(
                delta_type=event_type,
                item_id=_str_or_none(payload.get("item_id")),
                output_index=_index_or_none(payload.get("output_index")),
                call_id=_str_or_none(payload.get("call_id")),
            )
            self._deltas[slot] = buffer
        buffer.keys |= keys
        buffer.parts.append(delta)

    def _partial_items(self) -> list[PartialItem]:
        partial: list[PartialItem] = []
        for buffer in self._deltas.values():
            if buffer.keys & self._done_keys:
                continue  # the complete item arrived and is authoritative
            known = [key for key in sorted(buffer.keys, key=str) if key in self._added_items]
            added = self._added_items[known[0]] if known else None
            partial.append(
                PartialItem(
                    delta_type=buffer.delta_type,
                    item_id=buffer.item_id,
                    output_index=buffer.output_index,
                    call_id=buffer.call_id,
                    added_item=added,
                    text="".join(buffer.parts),
                )
            )
        return partial


def _completed_problem(response: Any) -> str | None:
    """Say why a response.completed payload is unusable, or return None if it is valid.

    Mirrors what Codex 0.164.0 requires: a ``response`` object with a string
    ``id`` and, if ``usage`` is present, integer token totals.
    """
    if not isinstance(response, dict):
        return "no 'response' object, so this is not a completion"
    if not isinstance(response.get("id"), str):
        return "response.id is not a string, so this is not a completion"
    usage = response.get("usage")
    if usage is not None:
        if not isinstance(usage, dict):
            return "response.usage is not an object, so this is not a completion"
        for key in _USAGE_TOTALS:
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                return f"response.usage.{key} is not an integer, so this is not a completion"
    return None


def _item_keys(payload: dict[str, Any], item: dict[str, Any] | None = None) -> set[_ItemKey]:
    """Every identifier this event gives for its output item."""
    keys: set[_ItemKey] = set()
    for value in (payload.get("item_id"), item.get("id") if item else None):
        if isinstance(value, str):
            keys.add(("id", value))
    index = _index_or_none(payload.get("output_index"))
    if index is not None:
        keys.add(("index", index))
    for value in (payload.get("call_id"), item.get("call_id") if item else None):
        if isinstance(value, str):
            keys.add(("call", value))
    return keys


def _primary_key(keys: set[_ItemKey]) -> _ItemKey:
    for kind in ("id", "index", "call"):
        for key in keys:
            if key[0] == kind:
                return key
    return ("none", None)


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _index_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
