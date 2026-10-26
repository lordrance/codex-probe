"""Integration tests for codex_probe.proxy: ProxyRecorder in front of the mock Responses server."""

from __future__ import annotations

import contextlib
import json
import re
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from codex_probe import ConfigError, LogWriteError, ProxyRecorder, RecorderError, load_session
from codex_probe.mock_server import (
    MockResponsesServer,
    ev_assistant_message,
    ev_completed,
    ev_exec_command_call,
    ev_response_created,
    json_response,
    raw_response,
    sse,
    sse_response,
)
from codex_probe.storage import (
    NO_SESSION_FILE,
    OTHER_REQUESTS_FILE,
    RUN_METADATA_FILE,
    decode_body,
    session_file_name,
    validate_call_record,
)

JSON = [("content-type", "application/json")]


def message_turn(response_id: str, text: str) -> list[dict[str, Any]]:
    return [ev_response_created(response_id), ev_assistant_message(f"msg_{response_id}", text), ev_completed(response_id)]


@pytest.fixture
def start_mock() -> Iterator[Callable[..., MockResponsesServer]]:
    """Start mock upstreams that are always stopped when the test ends."""
    servers: list[MockResponsesServer] = []

    def start(*script: Any) -> MockResponsesServer:
        server = MockResponsesServer(script)
        servers.append(server)
        server.start()
        return server

    try:
        yield start
    finally:
        for server in servers:
            server.stop()


@pytest.fixture
def start_recorder(tmp_path: Path) -> Iterator[Callable[..., ProxyRecorder]]:
    """Start ProxyRecorders that are always stopped when the test ends."""
    recorders: list[ProxyRecorder] = []

    def start(upstream: str, *, auth: dict[str, Any] | None = None, name: str = "mock", seed: int = 7) -> ProxyRecorder:
        recorder = ProxyRecorder(
            {
                "upstream": {"name": name, "base_url": upstream, "auth": auth or {"mode": "passthrough"}},
                "log_dir": str(tmp_path / "logs"),
                "seed": seed,
            }
        )
        recorders.append(recorder)
        recorder.start()
        return recorder

    try:
        yield start
    finally:
        for recorder in recorders:
            with contextlib.suppress(LogWriteError):
                recorder.stop()


@pytest.fixture
def client() -> Iterator[httpx.Client]:
    with httpx.Client(trust_env=False, timeout=60) as http:
        yield http


def send(
    client: httpx.Client,
    url: str,
    body: bytes = b"{}",
    headers: list[tuple[str, str]] | None = None,
    method: str = "POST",
) -> httpx.Response:
    """Send exactly these headers (plus Host and Content-Length, which HTTP requires)."""
    return client.send(httpx.Request(method, url, headers=headers if headers is not None else JSON, content=body))


def log_bytes(run_dir: Path) -> bytes:
    return b"".join(path.read_bytes() for path in sorted(run_dir.iterdir()) if path.is_file())


def wait_for_calls(recorder: ProxyRecorder, count: int, timeout: float = 10.0) -> list[dict[str, Any]]:
    """Read the run directory while the recorder runs, until ``count`` calls are on disk."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.suppress(Exception):
            calls = load_session(recorder.run_dir)
            if len(calls) >= count:
                return calls
        time.sleep(0.05)
    raise AssertionError(f"{count} call(s) were not logged within {timeout} s")


def mock_netloc(server: MockResponsesServer) -> str:
    assert server.base_url is not None
    return server.base_url.removeprefix("http://").removesuffix("/v1")


def assert_proxy_generated(recorded: dict[str, Any], received: httpx.Response, status: int) -> None:
    """The recorded response is, byte for byte, the proxy-generated one the client received."""
    assert recorded["source"] == "proxy"
    assert recorded["status"] == received.status_code == status
    assert recorded["headers"] == [[name.decode(), value.decode()] for name, value in received.headers.raw]
    assert decode_body(recorded["body"]) == received.content


# --- byte-for-byte passthrough ----------------------------------------------


def test_sse_call_is_relayed_byte_for_byte_and_recorded(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    events = [
        ev_response_created("resp_1"),
        ev_assistant_message("msg_1", "héllo 世界"),
        ev_exec_command_call("call_1", "python -m unittest"),
        ev_completed("resp_1"),
    ]
    mock = start_mock(sse_response(events))
    recorder = start_recorder(mock.base_url)
    body = json.dumps(
        {"model": "mock", "input": [], "client_metadata": {"session_id": "s-1", "turn_id": "t-1"}}
    ).encode()
    headers = [*JSON, ("accept", "text/event-stream"), ("session-id", "s-1"), ("thread-id", "th-1")]
    response = send(client, f"{recorder.endpoint}/responses", body, headers)
    exchange = mock.wait_for_exchanges(1)[0]
    assert exchange.request_body == body
    assert response.status_code == 200
    assert response.content == exchange.response_body == sse(events)
    [call] = recorder.stop()
    validate_call_record(call)
    assert call["seq"] == 0 and call["outcome"] == "completed"
    assert decode_body(call["request"]["body"]) == body
    assert decode_body(call["response"]["body"]) == sse(events)
    assert (call["session_key"], call["thread_id"], call["turn_id"]) == ("s-1", "th-1", "t-1")
    assert call["upstream"] == {
        "name": "mock",
        "url": f"{mock.base_url}/responses",
        "peer": mock_netloc(mock),
        "changes": ["request: Host set to the upstream host", "response: dropped hop-by-hop header 'transfer-encoding'"],
    }
    reconstruction = call["stream"]["reconstruction"]
    assert reconstruction["items"] == [events[1]["item"], events[2]["item"]]
    assert reconstruction["response_id"] == "resp_1"
    assert reconstruction["usage"] == events[-1]["response"]["usage"]
    assert [e["json"] for e in call["stream"]["events"]] == events
    assert call["derived"]["tool_calls"] == [{"type": "function_call", "name": "exec_command", "call_id": "call_1"}]
    assert call["response"]["source"] == "upstream" and call["proxy_response"] is None
    assert 0 <= call["timing"]["first_byte_ms"] <= call["timing"]["first_chunk_ms"] <= call["timing"]["total_ms"]


def test_plain_json_response_query_string_and_binary_request_body(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    reply = '{"id":"resp_9","output":[{"type":"message","content":"plain JSON, not a stream"}]}'.encode()
    mock = start_mock(raw_response([reply], headers=[*JSON, ("content-length", str(len(reply)))]))
    recorder = start_recorder(mock.base_url)
    body = b"\xff\x00 not UTF-8, not JSON"
    response = send(client, f"{recorder.endpoint}/responses?stream=false&path=%2Fa%20b", body)
    exchange = mock.wait_for_exchanges(1)[0]
    assert (exchange.path, exchange.query, exchange.request_body) == ("/v1/responses", "stream=false&path=%2Fa%20b", body)
    assert response.content == reply
    [call] = recorder.stop()
    assert call["outcome"] == "completed" and call["stream"] is None
    assert call["request"]["query"] == "stream=false&path=%2Fa%20b"
    assert "base64" in call["request"]["body"] and decode_body(call["request"]["body"]) == body
    assert decode_body(call["response"]["body"]) == reply
    assert call["upstream"]["url"] == f"{mock.base_url}/responses?stream=false&path=%2Fa%20b"


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
def test_upstream_http_errors_pass_through_unchanged(
    start_mock: Callable[..., MockResponsesServer],
    start_recorder: Callable[..., ProxyRecorder],
    client: httpx.Client,
    status: int,
) -> None:
    error = {"error": {"type": "upstream", "message": f"status {status} from the backend"}}
    mock = start_mock(json_response(status, error))
    recorder = start_recorder(mock.base_url)
    response = send(client, f"{recorder.endpoint}/responses")
    exchange = mock.wait_for_exchanges(1)[0]
    assert response.status_code == status
    assert response.content == exchange.response_body
    assert response.headers.raw == [(k.encode(), v.encode()) for k, v in exchange.response_headers]
    [call] = recorder.stop()
    assert call["outcome"] == "upstream_http_error"
    assert (call["response"]["source"], call["response"]["status"]) == ("upstream", status)
    assert decode_body(call["response"]["body"]) == exchange.response_body
    assert call["proxy_response"] is None


def test_expect_header_is_forwarded_unchanged(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    mock = start_mock(sse_response(message_turn("r", "ok")))
    recorder = start_recorder(mock.base_url)
    body = b'{"input": []}'
    response = send(client, f"{recorder.endpoint}/responses", body, [*JSON, ("expect", "100-continue")])
    assert response.status_code == 200 and response.content == sse(message_turn("r", "ok"))
    received = mock.wait_for_exchanges(1)[0].request_headers
    assert received == [
        ("host", mock_netloc(mock)),
        *JSON,
        ("expect", "100-continue"),
        ("content-length", str(len(body))),
    ]
    [call] = recorder.stop()
    assert call["outcome"] == "completed"
    assert not [change for change in call["upstream"]["changes"] if "expect" in change]


# --- headers -----------------------------------------------------------------


def test_request_headers_arrive_unchanged_except_the_allowed_changes(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    mock = start_mock(sse_response(message_turn("r", "ok")))
    recorder = start_recorder(mock.base_url)
    body = b'{"input": []}'
    sent = [
        *JSON,
        ("accept", "text/event-stream"),
        ("session-id", "s-1"),
        ("x-dup", "one"),
        ("x-dup", "two"),
        ("connection", "keep-alive, x-hop"),
        ("x-hop", "named by Connection"),
        ("keep-alive", "timeout=5"),
        ("te", "trailers"),
        ("proxy-authorization", "Basic cHJveHk="),
    ]
    send(client, f"{recorder.endpoint}/responses", body, sent)
    received = mock.wait_for_exchanges(1)[0].request_headers
    assert received == [
        ("host", mock_netloc(mock)),
        ("content-type", "application/json"),
        ("accept", "text/event-stream"),
        ("session-id", "s-1"),
        ("x-dup", "one"),
        ("x-dup", "two"),
        ("content-length", str(len(body))),
    ]
    for added in ("user-agent", "accept-encoding", "via", "x-forwarded-for", "x-forwarded-proto", "forwarded"):
        assert added not in {name for name, _ in received}
    [call] = recorder.stop()
    assert call["upstream"]["changes"][:6] == [
        "request: dropped hop-by-hop header 'connection'",
        "request: dropped hop-by-hop header 'x-hop'",
        "request: dropped hop-by-hop header 'keep-alive'",
        "request: dropped hop-by-hop header 'te'",
        "request: dropped hop-by-hop header 'proxy-authorization'",
        "request: Host set to the upstream host",
    ]


def test_response_headers_are_relayed_without_additions(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    reply = b'{"ok":true}'
    upstream_headers = [
        *JSON,
        ("x-request-id", "req_123"),
        ("set-cookie", "a=1"),
        ("set-cookie", "b=2"),
        ("date", "Fri, 23 Oct 2026 00:00:00 GMT"),
        ("server", "mock-upstream"),
        ("content-length", str(len(reply))),
    ]
    events = message_turn("r", "streamed")
    mock = start_mock(raw_response([reply], headers=upstream_headers), sse_response(events))
    recorder = start_recorder(mock.base_url)
    plain = send(client, f"{recorder.endpoint}/responses")
    # Exactly the upstream's headers: one Date and one Server, both from the upstream.
    assert plain.headers.raw == [(k.encode(), v.encode()) for k, v in upstream_headers]
    streamed = send(client, f"{recorder.endpoint}/responses")
    framing = [h for h in streamed.headers.raw if h[0].lower() == b"transfer-encoding"]
    assert framing == [(b"Transfer-Encoding", b"chunked")]  # framing added by our HTTP server (h11)
    assert [h for h in streamed.headers.raw if h[0].lower() != b"transfer-encoding"] == [
        (b"content-type", b"text/event-stream"),
        (b"cache-control", b"no-cache"),
    ]
    assert streamed.content == sse(events)
    calls = recorder.stop()
    assert calls[0]["response"]["headers"][2:4] == [["set-cookie", "[redacted]"], ["set-cookie", "[redacted]"]]


# --- auth modes --------------------------------------------------------------

CREDENTIALS = [
    ("authorization", "Bearer client-SECRET-1"),
    ("cookie", "c=SECRET-2"),
    ("chatgpt-account-id", "acct-SECRET-3"),
    ("x-api-key", "SECRET-4"),
]


def test_auth_passthrough_forwards_the_client_credentials(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    mock = start_mock(sse_response(message_turn("r", "ok")))
    recorder = start_recorder(mock.base_url, auth={"mode": "passthrough"})
    send(client, f"{recorder.endpoint}/responses", headers=[*JSON, *CREDENTIALS])
    received = mock.wait_for_exchanges(1)[0].request_headers
    assert [h for h in received if h in CREDENTIALS] == CREDENTIALS
    recorder.stop()
    assert b"SECRET" not in log_bytes(recorder.run_dir)


def test_auth_none_forwards_no_credentials(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    mock = start_mock(sse_response(message_turn("r", "ok")))
    recorder = start_recorder(mock.base_url, auth={"mode": "none"})
    send(client, f"{recorder.endpoint}/responses", headers=[*JSON, *CREDENTIALS])
    received = mock.wait_for_exchanges(1)[0].request_headers
    assert received == [("host", mock_netloc(mock)), *JSON, ("content-length", "2")]
    [call] = recorder.stop()
    assert [c for c in call["upstream"]["changes"] if "credential" in c] == [
        f"request: dropped credential header {name!r} (auth mode none)" for name, _ in CREDENTIALS
    ]
    assert b"SECRET" not in log_bytes(recorder.run_dir)


def test_auth_env_replaces_client_credentials_with_the_configured_key(
    start_mock: Callable[..., MockResponsesServer],
    start_recorder: Callable[..., ProxyRecorder],
    client: httpx.Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEXPROBE_TEST_KEY", "env-SECRET-key")
    mock = start_mock(sse_response(message_turn("r", "ok")))
    recorder = start_recorder(mock.base_url, auth={"mode": "env", "env_var": "CODEXPROBE_TEST_KEY"})
    send(client, f"{recorder.endpoint}/responses", headers=[*JSON, *CREDENTIALS])
    received = mock.wait_for_exchanges(1)[0].request_headers
    assert received == [("host", mock_netloc(mock)), *JSON, ("content-length", "2"), ("authorization", "Bearer env-SECRET-key")]
    [call] = recorder.stop()
    assert "request: authorization set from $CODEXPROBE_TEST_KEY (auth mode env)" in call["upstream"]["changes"]
    assert b"SECRET" not in log_bytes(recorder.run_dir)


def test_auth_env_without_the_variable_fails_at_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEXPROBE_MISSING_KEY", raising=False)
    recorder = ProxyRecorder(
        {
            "upstream": {"base_url": "http://127.0.0.1:9/v1", "auth": {"mode": "env", "env_var": "CODEXPROBE_MISSING_KEY"}},
            "log_dir": str(tmp_path / "logs"),
        }
    )
    with pytest.raises(ConfigError, match="CODEXPROBE_MISSING_KEY is not set or is empty"):
        recorder.start()
    assert recorder.stop() == []
    assert not (tmp_path / "logs").exists()


# --- backend swap ------------------------------------------------------------


def test_backend_is_chosen_by_config_alone(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    backend_a = start_mock(sse_response(message_turn("a", "from backend A")))
    backend_b = start_mock(sse_response(message_turn("b", "from backend B")))
    recorder_a = start_recorder(backend_a.base_url, name="backend-a")
    recorder_b = start_recorder(backend_b.base_url, name="backend-b")
    reply_a = send(client, f"{recorder_a.endpoint}/responses")
    reply_b = send(client, f"{recorder_b.endpoint}/responses")
    assert reply_a.content == sse(message_turn("a", "from backend A"))
    assert reply_b.content == sse(message_turn("b", "from backend B"))
    [call_a], [call_b] = recorder_a.stop(), recorder_b.stop()
    assert (call_a["upstream"]["name"], call_a["upstream"]["peer"]) == ("backend-a", mock_netloc(backend_a))
    assert (call_b["upstream"]["name"], call_b["upstream"]["peer"]) == ("backend-b", mock_netloc(backend_b))
    assert len(backend_a.exchanges) == len(backend_b.exchanges) == 1


# --- streaming: live relay, slow upstreams, cuts ----------------------------


def test_events_are_relayed_live(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    events = [ev_response_created("r"), *(ev_assistant_message(f"m{i}", f"part {i}") for i in range(3)), ev_completed("r")]
    mock = start_mock(sse_response(events, event_delay=0.2))
    recorder = start_recorder(mock.base_url)
    arrivals: list[tuple[float, bytes]] = []
    with client.stream("POST", f"{recorder.endpoint}/responses", content=b"{}") as response:
        for chunk in response.iter_raw():
            arrivals.append((time.perf_counter(), chunk))  # same high-resolution clock as the mock
    sent_at = mock.wait_for_exchanges(1)[0].response_chunk_times
    assert b"".join(chunk for _, chunk in arrivals) == sse(events)
    assert len(arrivals) == len(events)  # every event arrived on its own
    for i in range(len(events) - 1):
        assert arrivals[i][0] < sent_at[i + 1], f"event {i} arrived only after event {i + 1} was sent"
    added_ms = sorted((arrival - sent) * 1000 for (arrival, _), sent in zip(arrivals, sent_at))
    print(f"\nlive relay: added latency per event (ms) = {[round(ms, 2) for ms in added_ms]}")
    assert added_ms[-1] < 150


def test_slow_upstream_is_not_cut_by_the_proxy(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    events = [ev_assistant_message("m", "after a long silence"), ev_completed("r")]
    mock = start_mock(sse_response(events, first_byte_delay=6.2, event_delay=6.2))
    recorder = start_recorder(mock.base_url)
    started = time.monotonic()
    response = send(client, f"{recorder.endpoint}/responses")
    elapsed = time.monotonic() - started
    assert response.status_code == 200 and response.content == sse(events)
    [call] = recorder.stop()
    timing = call["timing"]
    print(f"\nslow upstream: {elapsed:.2f} s end to end, timing = {timing}")
    assert call["outcome"] == "completed"
    # Both silences are longer than httpx's default 5 s read timeout, so a proxy with
    # that default would have cut the stream. (asyncio on Windows may wake a timer up
    # to one 15.6 ms clock tick early, so the bounds are 6 s, not 6.2 s.)
    assert timing["first_byte_ms"] >= 6000
    assert timing["total_ms"] - timing["first_chunk_ms"] >= 6000
    assert elapsed >= 12.0


def test_unreachable_upstream_is_a_recorded_proxy_error(
    start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    recorder = start_recorder(f"http://127.0.0.1:{closed_port}/v1")
    response = send(client, f"{recorder.endpoint}/responses")
    assert response.status_code == 502
    error = response.json()["error"]
    assert error["type"] == "codexprobe_proxy_error"
    assert error["note"] == "generated by the CodexProbe proxy, not by the upstream"
    [call] = recorder.stop()
    assert call["outcome"] == "proxy_error"
    assert_proxy_generated(call["response"], response, 502)
    assert call["proxy_response"]["reason"].startswith("the upstream could not be reached: ConnectError")
    assert call["upstream"]["peer"] is None


def test_connect_timeout_is_a_recorded_504(
    start_recorder: Callable[..., ProxyRecorder], client: httpx.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def connect_times_out(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated connect timeout", request=request)

    # Only the proxy uses httpx.AsyncClient; the test client is the synchronous httpx.Client.
    monkeypatch.setattr(httpx.AsyncClient, "send", connect_times_out)
    recorder = start_recorder("http://127.0.0.1:9/v1")
    response = send(client, f"{recorder.endpoint}/responses")
    assert response.status_code == 504
    assert response.json()["error"]["type"] == "codexprobe_proxy_error"
    [call] = recorder.stop()
    assert call["outcome"] == "proxy_error"
    assert_proxy_generated(call["response"], response, 504)
    assert call["proxy_response"] == {
        "reason": "connecting to the upstream timed out (simulated connect timeout)"
    }


def test_stream_cut_before_the_terminal_event(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    partial = [ev_response_created("r"), ev_assistant_message("m", "no completed follows")]
    mock = start_mock(sse_response(partial), sse_response(partial, end="abort"))
    recorder = start_recorder(mock.base_url)
    clean = send(client, f"{recorder.endpoint}/responses")
    assert clean.status_code == 200 and clean.content == sse(partial)
    with client.stream("POST", f"{recorder.endpoint}/responses", content=b"{}") as response:
        with pytest.raises(httpx.RemoteProtocolError):
            response.read()  # the proxy passes the dropped connection on
    calls = recorder.stop()
    assert [c["outcome"] for c in calls] == ["stream_cut", "stream_cut"]
    assert [decode_body(c["response"]["body"]) for c in calls] == [sse(partial), sse(partial)]
    assert calls[0]["errors"]["upstream"] is None
    assert calls[1]["errors"]["upstream"].startswith("RemoteProtocolError")
    assert calls[1]["stream"]["reconstruction"]["items"] == [partial[1]["item"]]


def test_client_disconnect_before_completed_closes_the_upstream(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    events = message_turn("r", "never read")
    mock = start_mock(sse_response(events, event_delay=1.0))
    recorder = start_recorder(mock.base_url)
    with client.stream("POST", f"{recorder.endpoint}/responses", content=b"{}") as response:
        first = next(response.iter_raw())
    # Leaving the block closed the connection before the second event was sent.
    [call] = wait_for_calls(recorder, 1)
    assert first == sse(events[:1])
    assert call["outcome"] == "client_disconnected"
    assert decode_body(call["response"]["body"]) == first
    assert call["timing"]["total_ms"] < 900  # closed at once, not after the next upstream event
    # The upstream itself saw the proxy hang up while it waited to send the next event.
    [exchange] = mock.wait_for_exchanges(1)
    assert exchange.state == "peer_gone" and exchange.response_chunks == [first]


def test_client_disconnect_while_waiting_for_upstream_headers(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder]
) -> None:
    mock = start_mock(sse_response(message_turn("r", "never sent"), first_byte_delay=60))
    recorder = start_recorder(mock.base_url)
    port = int(recorder.endpoint.removeprefix("http://127.0.0.1:").removesuffix("/v1"))
    request = (
        b"POST /v1/responses HTTP/1.1\r\nhost: 127.0.0.1\r\ncontent-type: application/json\r\n"
        b"content-length: 2\r\n\r\n{}"
    )
    with socket.create_connection(("127.0.0.1", port), timeout=10) as raw:
        raw.sendall(request)
        deadline = time.monotonic() + 10
        while not mock.exchanges and time.monotonic() < deadline:
            time.sleep(0.02)
        assert mock.exchanges, "the request never reached the upstream"
    # The socket is now closed, about 59 s before the upstream would answer.
    hung_up = time.monotonic()
    [call] = wait_for_calls(recorder, 1)
    assert time.monotonic() - hung_up < 5  # recorded at once, not after the 60 s wait
    assert call["outcome"] == "client_disconnected"
    assert call["response"] == {"source": None, "status": None, "headers": None, "body": None}
    assert call["timing"]["first_byte_ms"] is None
    # The proxy cancelled its upstream request: the mock saw that connection close.
    [exchange] = mock.wait_for_exchanges(1, timeout=10)
    assert exchange.state == "peer_gone" and exchange.status is None
    print(f"\nclient left before the upstream answered: call recorded {call['timing']['total_ms']} ms after it arrived")


def test_client_disconnect_right_after_completed_is_still_completed(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    chunks = [*(sse([event]) for event in message_turn("r", "done")), b"data: [DONE]\n\n"]
    mock = start_mock(raw_response(chunks, chunk_delay=0.3))
    recorder = start_recorder(mock.base_url)
    received = b""
    with client.stream("POST", f"{recorder.endpoint}/responses", content=b"{}") as response:
        for chunk in response.iter_raw():
            received += chunk
            if b"response.completed" in received and received.endswith(b"\n\n"):
                break  # hang up like Codex does
    assert received == b"".join(chunks[:3])
    [call] = wait_for_calls(recorder, 1)
    assert call["outcome"] == "completed"
    assert decode_body(call["response"]["body"]) == b"".join(chunks)  # the trailing bytes were still kept


# OpenAI's ChatGPT Codex backend answers a streamed POST /v1/responses with no
# Content-Type header at all (VALIDATION.md, Phase 3).
STREAM_REQUEST = json.dumps({"model": "mock", "input": [], "stream": True}).encode()


def test_a_stream_without_content_type_is_parsed_when_the_request_asked_for_one(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    events = message_turn("r", "no content type")
    # A trailing SSE comment, sent late, so the client hangs up before the upstream ends.
    chunks = [*(sse([event]) for event in events), b": keep-alive\n\n"]
    mock = start_mock(raw_response(chunks, headers=[("cache-control", "no-cache")], chunk_delay=0.3))
    recorder = start_recorder(mock.base_url)
    request = httpx.Request("POST", f"{recorder.endpoint}/responses", headers=JSON, content=STREAM_REQUEST)
    received = b""
    response = client.send(request, stream=True)
    try:
        for chunk in response.iter_raw():
            received += chunk
            if b"response.completed" in received and received.endswith(b"\n\n"):
                break  # hang up like Codex does
    finally:
        response.close()
    assert received == b"".join(chunks[:3])
    assert [name for name, _ in response.headers.raw if name.lower() == b"content-type"] == []  # none was added
    [exchange] = mock.wait_for_exchanges(1)
    assert exchange.request_body == STREAM_REQUEST
    [call] = wait_for_calls(recorder, 1)
    assert decode_body(call["request"]["body"]) == STREAM_REQUEST
    assert decode_body(call["response"]["body"]) == exchange.response_body == b"".join(chunks)
    assert all(name.lower() != "content-type" for name, _ in call["response"]["headers"])
    assert call["outcome"] == "completed"
    assert [e["json"] for e in call["stream"]["events"]] == events
    reconstruction = call["stream"]["reconstruction"]
    assert reconstruction["terminal_event"] == "response.completed"
    assert reconstruction["items"] == [events[1]["item"]]
    assert call["derived"]["item_types"] == ["message"]


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        pytest.param(json.dumps({"stream": False}).encode(), [], id="stream false"),
        pytest.param(json.dumps({"stream": "true"}).encode(), [], id="stream the string true"),
        pytest.param(json.dumps({"stream": 1}).encode(), [], id="stream the number 1"),
        pytest.param(json.dumps([{"stream": True}]).encode(), [], id="a JSON array"),
        pytest.param(b'{"stream": true', [], id="invalid JSON"),
        pytest.param(STREAM_REQUEST, [("content-type", "application/octet-stream")], id="another content type"),
    ],
)
def test_without_both_signals_a_response_is_not_parsed_as_a_stream(
    start_mock: Callable[..., MockResponsesServer],
    start_recorder: Callable[..., ProxyRecorder],
    client: httpx.Client,
    body: bytes,
    headers: list[tuple[str, str]],
) -> None:
    payload = sse(message_turn("r", "not parsed"))
    mock = start_mock(raw_response([payload], headers=headers))
    recorder = start_recorder(mock.base_url)
    response = send(client, f"{recorder.endpoint}/responses", body)
    assert response.content == payload
    [call] = recorder.stop()
    assert call["stream"] is None and call["derived"]["event_count"] is None
    assert decode_body(call["request"]["body"]) == body and decode_body(call["response"]["body"]) == payload


def test_stop_while_waiting_for_the_upstream(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder]
) -> None:
    mock = start_mock(sse_response(message_turn("r", "too late"), first_byte_delay=60))
    recorder = start_recorder(mock.base_url)
    outcome: list[Any] = []

    def call() -> None:
        try:
            with httpx.Client(trust_env=False, timeout=60) as http:
                outcome.append(send(http, f"{recorder.endpoint}/responses"))
        except httpx.HTTPError as exc:
            outcome.append(exc)

    caller = threading.Thread(target=call)
    caller.start()
    deadline = time.monotonic() + 10
    while not mock.exchanges and time.monotonic() < deadline:
        time.sleep(0.02)
    assert mock.exchanges, "the request never reached the upstream"
    stopping = time.monotonic()
    [record] = recorder.stop()
    assert time.monotonic() - stopping < 10
    caller.join(15)
    [received] = outcome
    assert isinstance(received, httpx.Response) and received.status_code == 503
    assert record["outcome"] == "recorder_stopped"
    assert_proxy_generated(record["response"], received, 503)
    assert record["proxy_response"] == {"reason": "CodexProbe stopped before the upstream answered"}
    [exchange] = mock.wait_for_exchanges(1, timeout=10)
    assert exchange.state == "peer_gone"  # the upstream request was abandoned, not left waiting


def test_stop_in_the_middle_of_a_stream(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder]
) -> None:
    events = message_turn("r", "cut by stop")
    mock = start_mock(sse_response(events, event_delay=60))
    recorder = start_recorder(mock.base_url)
    got_first = threading.Event()
    seen: list[Any] = []

    def read() -> None:
        try:
            with httpx.Client(trust_env=False, timeout=60) as http:
                with http.stream("POST", f"{recorder.endpoint}/responses", content=b"{}") as response:
                    chunks = response.iter_raw()
                    seen.append(next(chunks))
                    got_first.set()
                    seen.extend(chunks)
        except httpx.HTTPError as exc:
            seen.append(exc)
        finally:
            got_first.set()

    reader = threading.Thread(target=read)
    reader.start()
    assert got_first.wait(10)
    stopping = time.monotonic()
    [record] = recorder.stop()
    assert time.monotonic() - stopping < 10
    reader.join(15)
    assert seen[0] == sse(events[:1])
    assert isinstance(seen[-1], httpx.RemoteProtocolError)  # the client sees the cut
    assert record["outcome"] == "recorder_stopped"
    assert decode_body(record["response"]["body"]) == sse(events[:1])


# --- lifecycle ---------------------------------------------------------------


def test_lifecycle_repeated_stop_and_port_release(start_mock: Callable[..., MockResponsesServer], tmp_path: Path) -> None:
    mock = start_mock()
    recorder = ProxyRecorder({"upstream": {"base_url": mock.base_url}, "log_dir": str(tmp_path / "logs")})
    try:
        assert recorder.stop() == []  # before start: nothing to stop
        endpoint = recorder.start()
        assert re.fullmatch(r"http://127\.0\.0\.1:\d+/v1", endpoint) and recorder.endpoint == endpoint
        with pytest.raises(RecorderError, match="only once"):
            recorder.start()
        with httpx.Client(trust_env=False, timeout=10) as http:
            assert http.get(f"{endpoint}/models").json() == {"models": []}
        calls = recorder.stop()
        assert calls == [] and recorder.stop() is calls
        port = int(endpoint.removeprefix("http://127.0.0.1:").removesuffix("/v1"))
        with socket.socket() as rebind:
            rebind.bind(("127.0.0.1", port))  # the port was released
        with pytest.raises(RecorderError, match="only once"):
            recorder.start()
    finally:
        recorder.stop()


def test_busy_port_is_a_clear_start_error(tmp_path: Path) -> None:
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        recorder = ProxyRecorder(
            {
                "upstream": {"base_url": "http://127.0.0.1:9/v1"},
                "listen": {"port": holder.getsockname()[1]},
                "log_dir": str(tmp_path / "logs"),
            }
        )
        with pytest.raises(RecorderError, match="could not start: cannot listen on 127.0.0.1"):
            recorder.start()
    assert not (tmp_path / "logs").exists()


def test_paths_outside_v1_are_refused_and_logged(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    mock = start_mock()
    recorder = start_recorder(mock.base_url)
    root = recorder.endpoint.removesuffix("/v1")
    refused = client.get(f"{root}/admin")
    assert refused.status_code == 404 and refused.json()["error"]["type"] == "codexprobe_proxy_error"
    assert recorder.stop() == []
    assert mock.exchanges == []
    [logged] = [json.loads(line) for line in (recorder.run_dir / OTHER_REQUESTS_FILE).read_text("utf-8").splitlines()]
    assert (logged["path"], logged["outcome"]) == ("/admin", "proxy_error")
    assert_proxy_generated(logged["response"], refused, 404)
    assert logged["proxy_response"] == {"reason": "CodexProbe forwards only paths under /v1/"}


# --- the logs ----------------------------------------------------------------


def test_calls_sessions_metadata_and_load_session(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    mock = start_mock(*(sse_response(message_turn(f"r{i}", f"answer {i}")) for i in range(3)))
    recorder = start_recorder(mock.base_url)
    secrets = [("authorization", "Bearer test-SECRET-1"), ("cookie", "c=SECRET-2")]
    for session in ("A", "B", None):
        headers = [*JSON, *secrets, *([("session-id", session)] if session else [])]
        send(client, f"{recorder.endpoint}/responses", headers=headers)
    models = client.get(f"{recorder.endpoint}/models", params={"client_version": "0.164.0"}, headers=dict(secrets))
    assert models.json() == {"models": []}
    calls = recorder.stop()
    run_dir = recorder.run_dir
    assert [c["seq"] for c in calls] == [0, 1, 2]
    assert [c["session_key"] for c in calls] == ["A", "B", None]
    for call in calls:
        validate_call_record(call)
    assert sorted(p.name for p in run_dir.iterdir()) == sorted(
        [session_file_name("A"), session_file_name("B"), NO_SESSION_FILE, OTHER_REQUESTS_FILE, RUN_METADATA_FILE]
    )
    assert load_session(run_dir) == calls
    [other] = [json.loads(line) for line in (run_dir / OTHER_REQUESTS_FILE).read_text("utf-8").splitlines()]
    assert (other["method"], other["path"], other["query"], other["outcome"]) == (
        "GET",
        "/v1/models",
        "client_version=0.164.0",
        "completed",
    )
    assert (other["response"]["source"], other["response"]["status"]) == ("upstream", 200)
    assert decode_body(other["response"]["body"]) == b'{"models":[]}'
    metadata = json.loads((run_dir / RUN_METADATA_FILE).read_text("utf-8"))
    assert (metadata["seed"], metadata["llm_calls"], metadata["other_requests"]) == (7, 3, 1)
    assert metadata["endpoint"] == recorder.endpoint and metadata["finished_at"] >= metadata["started_at"]
    assert metadata["config"]["upstream"]["base_url"] == mock.base_url
    assert {"python", "codex_probe", "httpx", "uvicorn"} <= set(metadata["versions"])
    assert b"SECRET" not in log_bytes(run_dir)


def test_call_ids_depend_only_on_seed_and_seq(
    start_mock: Callable[..., MockResponsesServer], start_recorder: Callable[..., ProxyRecorder], client: httpx.Client
) -> None:
    def call_ids(seed: int) -> list[str]:
        mock = start_mock(*(sse_response(message_turn(f"r{i}", "x")) for i in range(2)))
        recorder = start_recorder(mock.base_url, seed=seed)
        for _ in range(2):
            send(client, f"{recorder.endpoint}/responses")
        return [call["call_id"] for call in recorder.stop()]

    first, again, other = call_ids(7), call_ids(7), call_ids(8)
    assert first == again and len(set(first)) == 2
    assert set(first).isdisjoint(other)
    assert all(re.fullmatch(r"call_[0-9a-f]{16}", call_id) for call_id in first + other)
