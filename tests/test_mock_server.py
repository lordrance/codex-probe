"""Tests for codex_probe.mock_server: fixtures, scripting, timing, chunking, capture, lifecycle."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from codex_probe.mock_server import (
    Exchange,
    FunctionCall,
    MockResponsesServer,
    MockServerError,
    ev_apply_patch_exec_command_call,
    ev_assistant_message,
    ev_completed,
    ev_exec_command_call,
    ev_failed,
    ev_response_created,
    json_response,
    load_capture,
    raw_response,
    sse,
    sse_response,
)
from codex_probe.streaming import ResponseStreamAssembler, StreamResult

# The exact SSE that Codex 0.164.0's own test helpers produce for these events
# (codex-rs/core/tests/common/responses.rs, tag rust-v0.164.0).
CODEX_MESSAGE_SSE = (
    'event: response.created\n'
    'data: {"type":"response.created","response":{"id":"resp-1"}}\n'
    '\n'
    'event: response.output_item.done\n'
    'data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","id":"msg-1","content":[{"type":"output_text","text":"hi"}]}}\n'
    '\n'
    'event: response.completed\n'
    'data: {"type":"response.completed","response":{"id":"resp-1","usage":{"input_tokens":0,"input_tokens_details":null,"output_tokens":0,"output_tokens_details":null,"total_tokens":0}}}\n'
    '\n'
).encode()
CODEX_EXEC_COMMAND_EVENT = (
    'event: response.output_item.done\n'
    'data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call-1","name":"exec_command","arguments":"{\\"cmd\\":\\"echo hello\\"}"}}\n'
    '\n'
).encode()


@pytest.fixture
def start_mock() -> Iterator[Callable[..., MockResponsesServer]]:
    """Start mock servers that are always stopped when the test ends."""
    servers: list[MockResponsesServer] = []

    def start(*script: Any, **kwargs: Any) -> MockResponsesServer:
        server = MockResponsesServer(script, **kwargs)
        servers.append(server)
        server.start()
        return server

    try:
        yield start
    finally:
        for server in servers:
            server.stop()


@pytest.fixture
def client() -> Iterator[httpx.Client]:
    with httpx.Client(trust_env=False, timeout=20) as http:
        yield http


def assemble(body: bytes) -> StreamResult:
    assembler = ResponseStreamAssembler()
    assembler.feed(body)
    return assembler.finish()


def message_turn(response_id: str, text: str) -> list[dict[str, Any]]:
    return [ev_response_created(response_id), ev_assistant_message(f"msg-{response_id}", text), ev_completed(response_id)]


# --- fixtures match Codex's own ----------------------------------------------


def test_event_builders_reproduce_codex_fixtures_byte_for_byte() -> None:
    assert sse([ev_response_created("resp-1"), ev_assistant_message("msg-1", "hi"), ev_completed("resp-1")]) == CODEX_MESSAGE_SSE
    assert sse([ev_exec_command_call("call-1", "echo hello")]) == CODEX_EXEC_COMMAND_EVENT
    assert ev_exec_command_call("call-1", "echo hello")["item"] == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "exec_command",
        "arguments": '{"cmd":"echo hello"}',
    }
    assert json.loads(ev_apply_patch_exec_command_call("c", "*** Begin Patch\n*** End Patch")["item"]["arguments"]) == {
        "cmd": "apply_patch <<'EOF'\n*** Begin Patch\n*** End Patch\nEOF\n"
    }
    assert ev_failed("r", "context_length_exceeded", "too long") == {
        "type": "response.failed",
        "response": {"id": "r", "error": {"code": "context_length_exceeded", "message": "too long"}},
    }
    assert sse([{"type": "response.in_progress"}]) == b"event: response.in_progress\n\n"  # no data line


def test_message_stream_is_a_completed_response(
    start_mock: Callable[..., MockResponsesServer], client: httpx.Client
) -> None:
    server = start_mock(sse_response([ev_response_created("resp-1"), ev_assistant_message("msg-1", "hi"), ev_completed("resp-1")]))
    response = client.post(f"{server.base_url}/responses", json={"model": "mock", "input": []})
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.content == CODEX_MESSAGE_SSE
    result = assemble(response.content)
    assert result.completed and result.response_id == "resp-1"
    assert result.items == [ev_assistant_message("msg-1", "hi")["item"]]
    assert result.usage == ev_completed("resp-1")["response"]["usage"]


# --- routes ------------------------------------------------------------------


def test_models_websocket_probe_and_unknown_routes(
    start_mock: Callable[..., MockResponsesServer], client: httpx.Client
) -> None:
    server = start_mock()
    models = client.get(f"{server.base_url}/models", params={"client_version": "0.164.0"})
    assert models.status_code == 200 and models.json() == {"models": []}
    assert client.get(f"{server.base_url}/responses").status_code == 426
    unknown = client.post(f"{server.base_url}/chat/completions", json={})
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "mock_script_error"
    exchanges = server.wait_for_exchanges(3)
    assert [(e.method, e.path, e.query, e.status) for e in exchanges] == [
        ("GET", "/v1/models", "client_version=0.164.0", 200),
        ("GET", "/v1/responses", "", 426),
        ("POST", "/v1/chat/completions", "", 404),
    ]
    assert server.failures == ["no mock route for POST /v1/chat/completions"]


# --- scripted multi-turn conversations ---------------------------------------


def _answer(call_id: str, output: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def test_function_call_turn_then_answer(start_mock: Callable[..., MockResponsesServer], client: httpx.Client) -> None:
    server = start_mock(
        sse_response([ev_response_created("resp-1"), ev_exec_command_call("call-1", "python -m unittest"), ev_completed("resp-1")]),
        sse_response(message_turn("resp-2", "all tests passed")),
    )
    first = client.post(f"{server.base_url}/responses", json={"input": [{"type": "message", "role": "user", "content": []}]})
    assert assemble(first.content).items == [ev_exec_command_call("call-1", "python -m unittest")["item"]]
    second = client.post(
        f"{server.base_url}/responses",
        json={"input": [{"type": "function_call", "call_id": "call-1"}, _answer("call-1", "Process exited with code 0\nOutput:\nOK")]},
    )
    assert assemble(second.content).items[0]["content"][0]["text"] == "all tests passed"
    exchanges = server.wait_for_exchanges(2)
    assert exchanges[0].step == 0
    assert exchanges[0].function_calls == (FunctionCall("call-1", "exec_command", '{"cmd":"python -m unittest"}'),)
    assert exchanges[1].step == 1
    assert exchanges[1].function_call_outputs == {"call-1": "Process exited with code 0\nOutput:\nOK"}
    assert server.failures == []


@pytest.mark.parametrize(
    "bad_input",
    [[], [_answer("call-other", "x")], [{"type": "function_call", "call_id": "call-1"}]],
    ids=["no output", "wrong call_id", "call without output"],
)
def test_missing_function_call_output_is_refused_and_recorded(
    start_mock: Callable[..., MockResponsesServer], client: httpx.Client, bad_input: list[Any]
) -> None:
    server = start_mock(
        sse_response([ev_exec_command_call("call-1", "dir"), ev_completed("resp-1")]),
        sse_response(message_turn("resp-2", "done")),
    )
    client.post(f"{server.base_url}/responses", json={"input": []})
    refused = client.post(f"{server.base_url}/responses", json={"input": bad_input})
    assert refused.status_code == 400
    error = refused.json()["error"]
    assert error["code"] == "mock_script_error"
    assert error["message"] == "request 1 has no function_call_output for call_id(s) call-1 issued by script step 0"
    assert server.failures == [error["message"]]
    # The script did not move on: a correct request still gets step 1.
    answered = client.post(f"{server.base_url}/responses", json={"input": [_answer("call-1", "ok")]})
    assert answered.status_code == 200
    assert server.wait_for_exchanges(3)[2].step == 1


def test_exhausted_script_is_a_clear_error(start_mock: Callable[..., MockResponsesServer], client: httpx.Client) -> None:
    server = start_mock(sse_response(message_turn("resp-1", "only answer")))
    assert client.post(f"{server.base_url}/responses", json={"input": []}).status_code == 200
    extra = client.post(f"{server.base_url}/responses", json={"input": []})
    assert extra.status_code == 400
    message = "mock script exhausted: 1 response(s) were scripted and all have been used; request 1 has no answer"
    assert extra.json()["error"]["message"] == message
    assert server.failures == [message]
    server.add(sse_response(message_turn("resp-2", "added later")))
    assert assemble(client.post(f"{server.base_url}/responses", json={"input": []}).content).completed


# --- status, chunking, timing and cut streams ---------------------------------


def test_any_status_and_body(start_mock: Callable[..., MockResponsesServer], client: httpx.Client) -> None:
    server = start_mock(
        json_response(500, {"error": {"message": "upstream exploded"}}),
        raw_response([b"slow down"], status=429, headers=[("content-type", "text/plain"), ("retry-after", "2")]),
    )
    first = client.post(f"{server.base_url}/responses", json={})
    assert first.status_code == 500 and first.json() == {"error": {"message": "upstream exploded"}}
    second = client.post(f"{server.base_url}/responses", json={})
    assert second.status_code == 429
    assert second.headers["retry-after"] == "2"
    assert second.content == b"slow down"


def test_explicit_chunks_split_events_anywhere(start_mock: Callable[..., MockResponsesServer], client: httpx.Client) -> None:
    body = sse(message_turn("resp-1", "split 世界"))
    cut_inside_utf8 = body.index("世".encode()) + 1
    chunks = [body[:5], body[5:41], body[41:cut_inside_utf8], body[cut_inside_utf8:-1], body[-1:]]
    server = start_mock(raw_response(chunks))
    received = client.post(f"{server.base_url}/responses", json={}).content
    assert received == body
    assert assemble(received).items[0]["content"][0]["text"] == "split 世界"
    assert server.wait_for_exchanges(1)[0].response_chunks == chunks


def test_first_byte_delay_and_chunk_gaps_are_real(
    start_mock: Callable[..., MockResponsesServer], client: httpx.Client
) -> None:
    events = message_turn("resp-1", "timed")
    server = start_mock(sse_response(events, first_byte_delay=0.4, event_delay=0.25))
    started = time.monotonic()
    arrivals: list[float] = []
    received = b""
    with client.stream("POST", f"{server.base_url}/responses", json={}) as response:
        headers_at = time.monotonic() - started
        for chunk in response.iter_raw():
            arrivals.append(time.monotonic() - started)
            received += chunk
    assert headers_at >= 0.35
    assert received == sse(events)
    assert len(arrivals) == 3  # one chunk per event, each arriving on its own
    assert arrivals[1] - arrivals[0] >= 0.15 and arrivals[2] - arrivals[1] >= 0.15


def test_stream_that_ends_before_the_terminal_event(
    start_mock: Callable[..., MockResponsesServer], client: httpx.Client
) -> None:
    partial = [ev_response_created("resp-1"), ev_assistant_message("msg-1", "no completed follows")]
    server = start_mock(sse_response(partial), sse_response(partial, end="abort"))
    cleanly_ended = client.post(f"{server.base_url}/responses", json={})
    assert cleanly_ended.status_code == 200
    assert assemble(cleanly_ended.content).outcome == "stream_cut"
    with client.stream("POST", f"{server.base_url}/responses", json={}) as response:
        with pytest.raises(httpx.RemoteProtocolError):
            response.read()  # the connection is dropped before the chunked body ends
    exchanges = server.wait_for_exchanges(2)
    assert [e.state for e in exchanges] == ["sent", "aborted"]
    assert exchanges[1].response_body == sse(partial)


# --- capture -----------------------------------------------------------------


def _comparable(exchange: Exchange) -> dict[str, Any]:
    fields = dict(vars(exchange))
    fields.pop("request_headers")
    fields.pop("response_headers")
    return fields


def test_saved_capture_matches_memory_byte_for_byte_and_hides_credentials(
    start_mock: Callable[..., MockResponsesServer], client: httpx.Client, tmp_path: Path
) -> None:
    capture = tmp_path / "capture"
    server = start_mock(
        sse_response([ev_exec_command_call("call-1", "echo 世界"), ev_completed("resp-1")]),
        raw_response([b"\x00\xffbinary", b" tail"], status=502, headers=[("content-type", "application/octet-stream")]),
        sse_response([ev_response_created("resp-3")], end="abort"),
        capture_dir=capture,
    )
    secret_headers = {"Authorization": "Bearer sk-SECRET-1", "Cookie": "c=SECRET-2"}
    client.get(f"{server.base_url}/models", headers=secret_headers)
    client.post(f"{server.base_url}/responses?trace=1", content=b'{"input": []}', headers=secret_headers)
    client.post(f"{server.base_url}/responses", content=json.dumps({"input": [_answer("call-1", "世界")]}).encode())
    with pytest.raises(httpx.RemoteProtocolError):
        client.post(f"{server.base_url}/responses", content=b'{"input": [] }')
    in_memory = server.wait_for_exchanges(4)
    on_disk = load_capture(capture)
    assert [_comparable(e) for e in on_disk] == [_comparable(e) for e in in_memory]
    assert [e.request_body for e in on_disk] == [b"", b'{"input": []}', json.dumps({"input": [_answer("call-1", "世界")]}).encode(), b'{"input": [] }']
    assert on_disk[2].response_chunks == [b"\x00\xffbinary", b" tail"]
    assert on_disk[3].state == "aborted"
    # Credentials: kept in memory, never written to disk.
    assert ("authorization", "Bearer sk-SECRET-1") in in_memory[1].request_headers
    assert ("authorization", "[redacted]") in on_disk[1].request_headers
    assert all(b"SECRET" not in path.read_bytes() for path in capture.iterdir())
    assert server.capture_errors == []


# --- lifecycle ---------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_start_stop_restart_and_port_release() -> None:
    port = _free_port()
    server = MockResponsesServer(port=port)
    try:
        server.stop()  # before start: nothing to do
        assert server.start() == f"http://127.0.0.1:{port}/v1"
        with pytest.raises(MockServerError, match="already running"):
            server.start()
        with httpx.Client(trust_env=False, timeout=10) as http:
            assert http.get(f"{server.base_url}/models").status_code == 200
        server.stop()
        server.stop()  # a second stop does nothing and does not hang
        assert server.base_url is None
        # Windows retries a refused loopback connect for about 2 s, so a timeout is also a refusal.
        with httpx.Client(trust_env=False, timeout=5) as http:
            with pytest.raises((httpx.ConnectError, httpx.ConnectTimeout)):
                http.get(f"http://127.0.0.1:{port}/v1/models")
        with socket.socket() as rebind:
            rebind.bind(("127.0.0.1", port))  # the port was released
        assert server.start() == f"http://127.0.0.1:{port}/v1"  # and the server can start again
    finally:
        server.stop()


def test_busy_port_is_a_clear_start_error() -> None:
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        server = MockResponsesServer(port=holder.getsockname()[1])
        try:
            with pytest.raises(MockServerError, match="cannot listen on 127.0.0.1"):
                server.start()
        finally:
            server.stop()


def test_client_that_leaves_during_a_delay_is_noticed(start_mock: Callable[..., MockResponsesServer]) -> None:
    server = start_mock(sse_response(message_turn("resp-1", "never sent"), first_byte_delay=60))
    assert server.base_url is not None
    port = int(server.base_url.removeprefix("http://127.0.0.1:").removesuffix("/v1"))
    with socket.create_connection(("127.0.0.1", port), timeout=10) as raw:
        raw.sendall(b"POST /v1/responses HTTP/1.1\r\nhost: x\r\ncontent-length: 2\r\n\r\n{}")
        deadline = time.monotonic() + 10
        while not server.exchanges and time.monotonic() < deadline:
            time.sleep(0.01)
    hung_up = time.monotonic()
    [exchange] = server.wait_for_exchanges(1, timeout=10)
    assert time.monotonic() - hung_up < 5  # noticed at once, not after the 60 s delay
    assert exchange.state == "peer_gone"
    assert exchange.status is None and exchange.response_chunks == []


def test_stop_does_not_hang_on_a_response_in_progress(start_mock: Callable[..., MockResponsesServer]) -> None:
    server = start_mock(sse_response(message_turn("resp-1", "never sent"), first_byte_delay=60))
    outcome: list[Any] = []

    def call() -> None:
        try:
            with httpx.Client(trust_env=False, timeout=30) as http:
                outcome.append(http.post(f"{server.base_url}/responses", json={}).status_code)
        except httpx.HTTPError as exc:
            outcome.append(exc)

    caller = threading.Thread(target=call)
    caller.start()
    deadline = time.monotonic() + 5
    while not server.exchanges and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.exchanges and server.exchanges[0].state == "in_progress"
    stopping = time.monotonic()
    server.stop()
    assert time.monotonic() - stopping < 8
    caller.join(10)
    assert not caller.is_alive()
    assert outcome and outcome[0] != 200  # an error status or a dropped connection, never the response
    assert server.exchanges[0].state == "cancelled"
    assert server.exchanges[0].response_chunks == []
