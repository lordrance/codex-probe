"""Tests for codex_probe.streaming: SSE framing, chunking, and Responses stream reassembly."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from typing import Any

import pytest

from codex_probe.streaming import (
    PartialItem,
    ResponseStreamAssembler,
    SSEEvent,
    SSEParser,
    StreamResult,
)

# --- fixtures in the format of Codex's own test helpers (core/tests/common/responses.rs) ---

USAGE = {
    "input_tokens": 120,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 42,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 162,
}
MESSAGE = {
    "type": "message",
    "role": "assistant",
    "id": "msg_1",
    "status": "completed",
    "content": [{"type": "output_text", "text": "héllo 世界"}],
}
FUNCTION_CALL = {
    "type": "function_call",
    "call_id": "call_1",
    "name": "exec_command",
    "arguments": '{"cmd":"python -m unittest"}',
}
CUSTOM_TOOL_CALL = {
    "type": "custom_tool_call",
    "call_id": "call_2",
    "name": "apply_patch",
    "input": "*** Begin Patch\n*** Add File: a.py\n+print('hi')\n*** End Patch\n",
}
UNKNOWN_ITEM = {"type": "future_item_kind", "id": "fx_1", "payload": {"nested": [1, 2, 3]}}


def ev(event_type: str, **fields: Any) -> dict[str, Any]:
    """Build an event payload."""
    return {"type": event_type, **fields}


def frame(*events: dict[str, Any] | str) -> bytes:
    """Encode events like Codex's sse() helper: `event: <type>` then `data: <json>` then a blank line.

    A str argument is inserted verbatim (comments, [DONE], malformed input).
    """
    parts = []
    for event in events:
        if isinstance(event, str):
            parts.append(event)
        else:
            data = json.dumps(event, ensure_ascii=False)
            parts.append(f"event: {event['type']}\ndata: {data}\n\n")
    return "".join(parts).encode("utf-8")


def run(chunks: Iterable[bytes], end: str = "eof") -> StreamResult:
    """Feed the chunks to a fresh assembler and finish it."""
    assembler = ResponseStreamAssembler()
    for chunk in chunks:
        assembler.feed(chunk)
    return assembler.finish(end)  # type: ignore[arg-type]


FULL_EVENTS = [
    ev("response.created", response={"id": "resp_1"}),
    ev("response.output_item.added", output_index=0, item={**MESSAGE, "status": "in_progress", "content": []}),
    ev("response.output_text.delta", item_id="msg_1", output_index=0, content_index=0, delta="héllo "),
    ev("response.output_text.delta", item_id="msg_1", output_index=0, content_index=0, delta="世界"),
    ev("response.output_item.done", output_index=0, item=MESSAGE),
    ev("response.output_item.done", output_index=1, item=FUNCTION_CALL),
    ev("response.custom_tool_call_input.delta", call_id="call_2", output_index=2, delta="*** Begin Patch\n"),
    ev("response.output_item.done", output_index=2, item=CUSTOM_TOOL_CALL),
    ev("response.output_item.done", output_index=3, item=UNKNOWN_ITEM),
    ev("response.completed", response={"id": "resp_1", "usage": USAGE}),
]
FULL_STREAM = frame(
    ": stream opened\n\n",
    *FULL_EVENTS[:5],
    ": keep-alive\n\nid: 7\nretry: 1000\n",
    *FULL_EVENTS[5:],
)


# --- a complete stream -----------------------------------------------------


def test_complete_stream_is_reassembled_from_done_items() -> None:
    result = run([FULL_STREAM])
    assert result.completed
    assert result.outcome == "completed"
    assert result.terminal_event == "response.completed"
    assert result.response_id == "resp_1"
    assert result.usage == USAGE
    # Every done item, in output order, exactly as sent: message, both tool-call kinds, unknown kind.
    assert result.items == [MESSAGE, FUNCTION_CALL, CUSTOM_TOOL_CALL, UNKNOWN_ITEM]
    assert result.partial_items == []
    assert result.parse_errors == []
    assert result.comment_count == 2
    assert not result.ended_mid_event
    assert [e.type for e in result.events] == [e["type"] for e in FULL_EVENTS]
    assert [e.json for e in result.events] == FULL_EVENTS
    assert result.events[3].data == json.dumps(FULL_EVENTS[3], ensure_ascii=False)
    assert result.events[0].event == "response.created"
    assert result.events[4].id is None and result.events[5].id == "7"


def test_every_two_way_split_gives_the_same_result() -> None:
    expected = run([FULL_STREAM])
    for cut in range(1, len(FULL_STREAM)):
        assert run([FULL_STREAM[:cut], FULL_STREAM[cut:]]) == expected, f"cut at byte {cut}"


def test_one_byte_chunks_and_random_chunkings_give_the_same_result() -> None:
    expected = run([FULL_STREAM])
    assert run(FULL_STREAM[i : i + 1] for i in range(len(FULL_STREAM))) == expected
    rng = random.Random(1234)
    for _ in range(200):
        cuts = sorted(rng.sample(range(1, len(FULL_STREAM)), rng.randint(1, 40)))
        bounds = [0, *cuts, len(FULL_STREAM)]
        chunks = [FULL_STREAM[a:b] for a, b in zip(bounds, bounds[1:])]
        assert run(chunks) == expected, cuts


def test_utf8_character_split_across_chunks_is_decoded_intact() -> None:
    start = FULL_STREAM.index("世".encode("utf-8"))
    for cut in (start + 1, start + 2):  # inside the 3-byte encoding
        result = run([FULL_STREAM[:cut], FULL_STREAM[cut:]])
        assert result.events[3].json["delta"] == "世界"
        assert result.items[0]["content"][0]["text"] == "héllo 世界"
        assert "\ufffd" not in "".join(e.data for e in result.events)


@pytest.mark.parametrize("newline", [b"\r\n", b"\r"], ids=["CRLF", "CR"])
def test_crlf_and_cr_line_endings_parse_like_lf(newline: bytes) -> None:
    converted = FULL_STREAM.replace(b"\n", newline)
    expected = run([FULL_STREAM])
    assert run([converted]) == expected
    # A CR at the end of a chunk may be half of a CRLF: every split must still agree.
    for cut in range(1, len(converted)):
        assert run([converted[:cut], converted[cut:]]) == expected, f"cut at byte {cut}"


def test_mixed_line_endings_in_one_stream() -> None:
    stream = (
        b'data: {"type":"response.output_item.done","item":{"type":"message","id":"m"}}\r\n\r\n'
        b": ping\r\r"
        b'data: {"type":"response.completed","response":{"id":"r"}}\n\n'
    )
    result = run([stream])
    assert result.completed
    assert result.items == [{"type": "message", "id": "m"}]
    assert result.comment_count == 1


def test_multi_line_data_is_joined_with_newlines() -> None:
    stream = b'data: {"type": "response.completed",\ndata:  "response": {"id": "r"}}\n\n'
    result = run([stream])
    assert result.events[0].data == '{"type": "response.completed",\n "response": {"id": "r"}}'
    assert result.completed and result.response_id == "r"


def test_comments_and_heartbeats_dispatch_nothing() -> None:
    stream = b": ping\n\n:\n\n" + frame(FULL_EVENTS[-1]) + b": trailing ping\n\n"
    result = run([stream])
    assert result.comment_count == 3
    assert [e.type for e in result.events] == ["response.completed"]
    assert result.completed


def test_byte_order_mark_at_stream_start_is_ignored() -> None:
    result = run([b"\xef\xbb\xbf" + frame(FULL_EVENTS[-1])])
    assert result.events[0].event == "response.completed"
    assert result.completed


# --- terminal events ---------------------------------------------------------


def test_completed_without_terminating_blank_line_is_not_a_completion() -> None:
    completed = b'data: {"type":"response.completed","response":{"id":"resp_1"}}'
    for tail in (completed + b"\n", completed):
        result = run([frame(FULL_EVENTS[0], FULL_EVENTS[4]) + tail])
        assert result.outcome == "stream_cut"
        assert not result.completed
        assert result.terminal_event is None
        assert result.ended_mid_event
        assert result.items == [MESSAGE]
        assert result.parse_errors[-1].startswith("the stream ended in the middle of an event")


def test_completed_with_broken_json_is_not_a_completion() -> None:
    broken = '{"type":"response.completed","response":{"id":"resp_1"'
    result = run([frame(FULL_EVENTS[4], f"data: {broken}\n\n")])
    assert result.outcome == "stream_cut"
    assert result.events[-1].data == broken
    assert result.events[-1].json is None
    assert "data is not valid JSON" in result.parse_errors[0]


@pytest.mark.parametrize(
    ("response", "problem"),
    [
        (None, "no 'response' object"),
        ({"usage": USAGE}, "response.id is not a string"),
        ({"id": 7}, "response.id is not a string"),
        ({"id": "resp_1", "usage": "lots"}, "response.usage is not an object"),
        ({"id": "resp_1", "usage": {**USAGE, "total_tokens": "162"}}, "response.usage.total_tokens is not an integer"),
        ({"id": "resp_1", "usage": {**USAGE, "input_tokens": True}}, "response.usage.input_tokens is not an integer"),
    ],
)
def test_invalid_completed_is_recorded_and_a_later_valid_one_still_counts(
    response: Any, problem: str
) -> None:
    bad = ev("response.completed") if response is None else ev("response.completed", response=response)
    good = ev("response.completed", response={"id": "resp_2"})
    first = run([frame(bad)])
    assert first.outcome == "stream_cut" and first.terminal_event is None
    assert problem in first.parse_errors[0]
    second = run([frame(bad, good)])
    assert second.completed and second.response_id == "resp_2"
    assert problem in second.parse_errors[0]


def test_done_sentinel_after_completed_is_ignored() -> None:
    result = run([frame(*FULL_EVENTS, "data: [DONE]\n\n")])
    assert result.completed
    assert result.events[-1].data == "[DONE]"
    assert len(result.parse_errors) == 1 and "data is not valid JSON" in result.parse_errors[0]


def test_incomplete_records_reason_and_usage() -> None:
    incomplete = ev(
        "response.incomplete",
        response={
            "id": "resp_1",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": USAGE,
        },
    )
    result = run([frame(FULL_EVENTS[0], FULL_EVENTS[4], incomplete)])
    assert result.outcome == "incomplete"
    assert not result.completed
    assert result.terminal_event == "response.incomplete"
    assert result.incomplete_reason == "max_output_tokens"
    assert result.usage == USAGE
    assert result.items == [MESSAGE]
    assert result.error is None


def test_failed_records_the_error_object() -> None:
    error = {"code": "context_length_exceeded", "message": "Your input exceeds the context window"}
    failed = ev("response.failed", response={"id": "resp_1", "status": "failed", "error": error})
    result = run([frame(FULL_EVENTS[0], failed)])
    assert result.outcome == "failed"
    assert result.terminal_event == "response.failed"
    assert result.error == error
    assert result.response_id == "resp_1"


def test_error_event_with_top_level_fields() -> None:
    error_event = ev("error", code="server_error", message="boom", param=None, sequence_number=3)
    result = run([frame(FULL_EVENTS[0], error_event)])
    assert result.outcome == "failed"
    assert result.terminal_event == "error"
    assert result.error == {"code": "server_error", "message": "boom", "param": None}


def test_error_event_with_nested_error_object() -> None:
    nested = {"code": "flex_unavailable", "message": "try later"}
    result = run([frame(ev("error", error=nested))])
    assert result.outcome == "failed"
    assert result.error == nested


def test_events_after_the_terminal_event_do_not_change_the_result() -> None:
    late_item = ev("response.output_item.done", output_index=9, item={"type": "message", "id": "late"})
    result = run([frame(*FULL_EVENTS, late_item, ev("response.failed", response={"id": "x"}))])
    assert result.completed
    assert result.items == [MESSAGE, FUNCTION_CALL, CUSTOM_TOOL_CALL, UNKNOWN_ITEM]
    assert result.response_id == "resp_1"
    assert [e.json for e in result.events[-2:]] == [late_item, ev("response.failed", response={"id": "x"})]
    assert all("after the terminal event response.completed" in p for p in result.parse_errors)
    assert len(result.parse_errors) == 2


def test_client_disconnect_after_completed_is_still_a_completion() -> None:
    result = run([FULL_STREAM], end="client_disconnected")
    assert result.outcome == "completed"
    assert result.partial_items == []


# --- interrupted streams and deltas ------------------------------------------

MESSAGE_SKELETON = {"type": "message", "role": "assistant", "id": "msg_1", "status": "in_progress", "content": []}
CALL_SKELETON = {"type": "function_call", "id": "fc_9", "call_id": "call_9", "name": "exec_command", "arguments": ""}


@pytest.mark.parametrize(
    ("end", "outcome"),
    [("eof", "stream_cut"), ("client_disconnected", "client_disconnected"), ("recorder_stopped", "recorder_stopped")],
)
def test_interrupted_stream_rebuilds_partial_text_from_deltas(end: str, outcome: str) -> None:
    stream = frame(
        ev("response.created", response={"id": "resp_1"}),
        ev("response.output_item.added", output_index=0, item=MESSAGE_SKELETON),
        ev("response.output_text.delta", item_id="msg_1", output_index=0, content_index=0, delta="Hel"),
        ev("response.output_text.delta", item_id="msg_1", output_index=0, content_index=0, delta="lo 世"),
    )
    result = run([stream], end=end)
    assert result.outcome == outcome
    assert not result.completed
    assert result.items == []
    assert result.response_id == "resp_1"
    assert result.partial_items == [
        PartialItem(
            delta_type="response.output_text.delta",
            item_id="msg_1",
            output_index=0,
            call_id=None,
            added_item=MESSAGE_SKELETON,
            text="Hello 世",
        )
    ]


def test_done_item_wins_over_its_deltas_and_is_not_repeated() -> None:
    done_message = {**MESSAGE_SKELETON, "status": "completed", "content": [{"type": "output_text", "text": "Hello world"}]}
    stream = frame(
        ev("response.output_item.added", output_index=0, item=MESSAGE_SKELETON),
        ev("response.output_text.delta", item_id="msg_1", output_index=0, content_index=0, delta="Hello wor"),
        ev("response.output_item.done", output_index=0, item=done_message),
        ev("response.output_item.added", output_index=1, item=CALL_SKELETON),
        ev("response.function_call_arguments.delta", item_id="fc_9", output_index=1, delta='{"cmd":'),
        ev("response.function_call_arguments.delta", item_id="fc_9", output_index=1, delta='"dir"}'),
    )
    result = run([stream])
    assert result.outcome == "stream_cut"
    assert result.items == [done_message]  # from the done event, not from "Hello wor"
    assert result.partial_items == [
        PartialItem(
            delta_type="response.function_call_arguments.delta",
            item_id="fc_9",
            output_index=1,
            call_id=None,
            added_item=CALL_SKELETON,
            text='{"cmd":"dir"}',
        )
    ]


def test_custom_tool_input_deltas_keyed_by_call_id() -> None:
    stream = frame(
        ev("response.custom_tool_call_input.delta", call_id="call_2", delta="*** Begin Patch\n"),
        ev("response.custom_tool_call_input.delta", call_id="call_2", delta="*** Add File: a.py\n"),
    )
    result = run([stream], end="recorder_stopped")
    assert result.outcome == "recorder_stopped"
    assert result.partial_items == [
        PartialItem(
            delta_type="response.custom_tool_call_input.delta",
            item_id=None,
            output_index=None,
            call_id="call_2",
            added_item=None,
            text="*** Begin Patch\n*** Add File: a.py\n",
        )
    ]
    # Once the done item for that call arrives, the deltas are not reported as partial output.
    finished = run([stream + frame(ev("response.output_item.done", item=CUSTOM_TOOL_CALL))])
    assert finished.items == [CUSTOM_TOOL_CALL]
    assert finished.partial_items == []


def test_terminal_event_suppresses_partial_items() -> None:
    stream = frame(
        ev("response.output_item.added", output_index=0, item=MESSAGE_SKELETON),
        ev("response.output_text.delta", item_id="msg_1", output_index=0, content_index=0, delta="cut off"),
        ev("response.incomplete", response={"id": "r", "incomplete_details": {"reason": "max_output_tokens"}}),
    )
    result = run([stream])
    assert result.outcome == "incomplete"
    assert result.partial_items == []


# --- malformed events do not stop parsing ------------------------------------


def test_malformed_events_are_recorded_and_parsing_continues() -> None:
    stream = frame(
        "data: not json\n\n",
        "data: [1, 2]\n\n",
        'data: {"no_type": true}\n\n',
        ev("response.output_item.done", item="not an object"),
        ev("response.output_text.delta", item_id="m", delta=5),
        FULL_EVENTS[4],
        FULL_EVENTS[-1],
    )
    result = run([stream])
    assert result.completed
    assert result.items == [MESSAGE]
    assert [e.error is not None for e in result.events] == [True, True, True, True, True, False, False]
    # The first message ends with the json module's own wording, so only its start is fixed.
    assert result.parse_errors[0].startswith("event 0 (no type): data is not valid JSON")
    assert result.parse_errors[1:] == [
        "event 1 (no type): data is not a JSON object",
        "event 2 (no type): the JSON has no string 'type' field",
        "event 3 (response.output_item.done): no 'item' object",
        "event 4 (response.output_text.delta): no string 'delta'",
    ]
    assert result.events[1].json == [1, 2]


def test_finish_rejects_unknown_end_and_feed_after_finish() -> None:
    assembler = ResponseStreamAssembler()
    with pytest.raises(ValueError, match="end must be one of"):
        assembler.finish("timeout")  # type: ignore[arg-type]
    result = assembler.finish()
    assert assembler.finish("client_disconnected") is result
    with pytest.raises(RuntimeError, match="after finish"):
        assembler.feed(b"data: x\n\n")


# --- the SSE parser on its own -----------------------------------------------


def test_parser_field_rules() -> None:
    parser = SSEParser()
    events = parser.feed(
        b"event: first\n"
        b"data:no-space\n"
        b"data\n"  # a field name without a colon has an empty value
        b"id: 42\n"
        b"\n"
        b"event: no-data-so-not-dispatched\n"
        b"\n"
        b"data: second\n"
        b"\n"
    )
    assert events == [
        SSEEvent(index=0, event="first", data="no-space\n", id="42"),
        SSEEvent(index=1, event=None, data="second", id="42"),
    ]
    assert parser.close() == []
    assert not parser.ended_mid_event


def test_parser_holds_a_trailing_cr_until_the_next_chunk() -> None:
    parser = SSEParser()
    assert parser.feed(b"data: a\r\n\r") == []  # this CR may be half of a CRLF
    assert parser.feed(b"\n") == [SSEEvent(index=0, event=None, data="a", id=None)]
    assert parser.feed(b"data: b\n\r") == []
    assert parser.close() == [SSEEvent(index=1, event=None, data="b", id=None)]  # at EOF a lone CR ends the line
