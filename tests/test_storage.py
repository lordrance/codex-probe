"""Tests for codex_probe.storage: bodies, header redaction, the background writer and load_session."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import codex_probe
from codex_probe.storage import (
    NO_SESSION_FILE,
    OTHER_REQUESTS_FILE,
    REDACTED,
    RUN_METADATA_FILE,
    LogWriteError,
    RecordError,
    RunWriter,
    SessionLoadError,
    call_identity,
    decode_body,
    encode_body,
    load_session,
    redact_headers,
    session_file_name,
    validate_call_record,
)

SECRETS = {
    "Authorization": "Bearer sk-live-SECRET-1",
    "proxy-authorization": "Basic cHJveHk6U0VDUkVULTI=",
    "Cookie": "session=SECRET-3",
    "Set-Cookie": "__cf_bm=SECRET-4; path=/",
    "ChatGPT-Account-ID": "acct-SECRET-5",
    "x-api-key": "SECRET-6",
    "api-key": "SECRET-7",
    "OpenAI-Api-Key": "SECRET-8",
    "x-auth-token": "SECRET-9",
}


@pytest.fixture
def make_writer(tmp_path: Path) -> Iterator[Callable[..., RunWriter]]:
    """Create RunWriters that are always stopped when the test ends."""
    writers: list[RunWriter] = []

    def make(run_dir: Path | None = None) -> RunWriter:
        writer = RunWriter(run_dir if run_dir is not None else tmp_path / "run")
        writers.append(writer)
        return writer

    yield make
    for writer in writers:
        try:
            writer.stop(timeout=10)
        except LogWriteError:
            pass


def make_record(seq: int, session_key: str | None = "session-A", **overrides: Any) -> dict[str, Any]:
    """A valid call record, built with the module's own helpers."""
    request_body = json.dumps(
        {"model": "mock-model", "input": [{"role": "user", "content": "héllo 世界"}],
         "client_metadata": {"session_id": session_key}}
    ).encode("utf-8")
    response_body = b'data: {"type":"response.completed","response":{"id":"resp_1"}}\n\n'
    record: dict[str, Any] = {
        "kind": "llm_call",
        "seq": seq,
        "call_id": f"call-{seq}",
        "session_key": session_key,
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "upstream": {"name": "mock", "url": "http://127.0.0.1:9/v1/responses", "peer": "127.0.0.1:9"},
        "request": {
            "method": "POST",
            "path": "/v1/responses",
            "headers": redact_headers([("content-type", "application/json"), ("Authorization", SECRETS["Authorization"])]),
            "body": encode_body(request_body),
        },
        "response": {
            "source": "upstream",
            "status": 200,
            "headers": redact_headers([("content-type", "text/event-stream")]),
            "body": encode_body(response_body),
        },
        "stream": {
            "events": [{"index": 0, "type": "response.completed"}],
            "reconstruction": {"outcome": "completed", "items": [], "response_id": "resp_1"},
        },
        "timing": {"started_at": "2026-10-24T10:00:00.000Z", "first_byte_ms": 12.5, "total_ms": 40.0},
        "outcome": "completed",
        "derived": {"request_bytes": len(request_body), "response_bytes": len(response_body)},
    }
    record.update(overrides)
    return record


def all_bytes(directory: Path) -> bytes:
    """Every byte written under a directory."""
    return b"".join(p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file())


# --- bodies ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ("héllo 世界".encode("utf-8"), {"text": "héllo 世界"}),
        (b"\xff\x00\xfe binary", {"base64": "/wD+IGJpbmFyeQ=="}),
        (b'{"a": [1, 2], "b": "\xe4\xb8\x96"}', {"text": '{"a": [1, 2], "b": "世"}', "json": {"a": [1, 2], "b": "世"}}),
        (b"", {"text": ""}),
        (b"[1, NaN]", {"text": "[1, NaN]"}),  # not standard JSON, so no json view
        (b"\xef\xbb\xbf{}", {"text": "\ufeff{}"}),  # a BOM is kept as text
    ],
)
def test_encode_body_is_lossless(data: bytes, expected: dict[str, Any]) -> None:
    body = encode_body(data)
    assert body == expected
    assert decode_body(body) == data


def test_random_bytes_round_trip() -> None:
    rng = random.Random(7)
    for size in (1, 2, 3, 17, 4096):
        for _ in range(20):
            data = bytes(rng.randrange(256) for _ in range(size))
            assert decode_body(encode_body(data)) == data


# --- headers -----------------------------------------------------------------


def test_redaction_keeps_order_and_duplicates_and_hides_every_credential() -> None:
    headers = [
        ("Content-Type", "application/json"),
        *SECRETS.items(),
        ("Set-Cookie", "second=SECRET-10"),
        ("x-ratelimit-remaining-tokens", "1000"),  # a count, not a credential
        (b"session-id", b"abc"),
    ]
    assert redact_headers(headers) == [
        ["Content-Type", "application/json"],
        *[[name, REDACTED] for name in SECRETS],
        ["Set-Cookie", REDACTED],
        ["x-ratelimit-remaining-tokens", "1000"],
        ["session-id", "abc"],
    ]


def test_credentials_never_reach_the_disk(make_writer: Callable[..., RunWriter], tmp_path: Path) -> None:
    writer = make_writer()
    secret_headers = [*SECRETS.items(), ("Set-Cookie", "second=SECRET-10")]
    record = make_record(0)
    record["request"]["headers"] = redact_headers(secret_headers)
    record["response"]["headers"] = redact_headers(secret_headers)
    writer.submit_call(record)
    writer.submit_request_log(
        {"kind": "request", "method": "GET", "path": "/v1/models", "headers": redact_headers(secret_headers)}
    )
    writer.stop()
    written = all_bytes(tmp_path / "run")
    for secret in [*SECRETS.values(), "second=SECRET-10"]:
        assert secret.encode() not in written
    assert b"SECRET" not in written
    assert written.count(REDACTED.encode()) == 3 * len(secret_headers)


def test_unredacted_credential_header_is_rejected_before_queueing(
    make_writer: Callable[..., RunWriter], tmp_path: Path
) -> None:
    writer = make_writer()
    record = make_record(0)
    record["request"]["headers"] = [["authorization", "Bearer sk-live-SECRET-1"]]
    with pytest.raises(RecordError, match="credential header 'authorization' must be redacted") as info:
        writer.submit_call(record)
    assert "SECRET" not in str(info.value)
    with pytest.raises(RecordError, match="must be redacted"):
        writer.submit_request_log({"kind": "request", "headers": [["Cookie", "a=SECRET"]]})
    assert writer.stop() == []
    assert not (tmp_path / "run").exists() or b"SECRET" not in all_bytes(tmp_path / "run")


# --- session identity and files ----------------------------------------------


def test_call_identity_prefers_headers_then_client_metadata() -> None:
    body = {"client_metadata": {"session_id": "s-body", "thread_id": "t-body", "turn_id": "turn-body"}}
    assert call_identity([("Session-Id", "s-head"), ("thread-id", "t-head")], body) == {
        "session_key": "s-head",
        "thread_id": "t-head",
        "turn_id": "turn-body",
    }
    assert call_identity([], body) == {"session_key": "s-body", "thread_id": "t-body", "turn_id": "turn-body"}
    turn_header = json.dumps({"turn_id": "turn-head"})
    assert call_identity([(b"x-codex-turn-metadata", turn_header.encode())], None) == {
        "session_key": None,
        "thread_id": None,
        "turn_id": "turn-head",
    }
    assert call_identity([("session-id", "")], {"client_metadata": "not a map"}) == {
        "session_key": None,
        "thread_id": None,
        "turn_id": None,
    }


def test_each_session_gets_its_own_file_and_a_stable_name(
    make_writer: Callable[..., RunWriter], tmp_path: Path
) -> None:
    writer = make_writer()
    for seq, key in enumerate(["A", "B", "A", None, "B", "A"]):
        writer.submit_call(make_record(seq, key))
    writer.stop()
    run = tmp_path / "run"
    name_a = "session-" + hashlib.sha256(b"A").hexdigest() + ".jsonl"
    assert session_file_name("A") == name_a
    assert sorted(p.name for p in run.iterdir()) == sorted([name_a, session_file_name("B"), NO_SESSION_FILE])
    seqs = {p.name: [json.loads(line)["seq"] for line in p.read_text("utf-8").splitlines()] for p in run.iterdir()}
    assert seqs == {name_a: [0, 2, 5], session_file_name("B"): [1, 4], NO_SESSION_FILE: [3]}


@pytest.mark.parametrize(
    "key", ["../../escaped", "..\\..\\escaped", "C:\\Windows\\escaped", "/etc/escaped", "con", "a\x00b", "\ud800"]
)
def test_hostile_session_keys_cannot_choose_a_path(
    make_writer: Callable[..., RunWriter], tmp_path: Path, key: str
) -> None:
    writer = make_writer()
    writer.submit_call(make_record(0, key))
    writer.stop()
    run = tmp_path / "run"
    assert [p.name for p in tmp_path.iterdir()] == ["run"]
    assert [p.name for p in run.iterdir()] == [session_file_name(key)]
    assert session_file_name(key).startswith("session-") and len(session_file_name(key)) == len("session-") + 64 + len(".jsonl")
    assert load_session(run)[0]["session_key"] == key


# --- writing and reading back ------------------------------------------------


def test_concurrent_submits_load_back_in_seq_order(make_writer: Callable[..., RunWriter], tmp_path: Path) -> None:
    writer = make_writer()
    seqs = list(range(200))
    random.Random(3).shuffle(seqs)
    batches = [seqs[i::8] for i in range(8)]
    keys = ["A", "B", None, "C"]

    def submit(batch: list[int]) -> None:
        for seq in batch:
            writer.submit_call(make_record(seq, keys[seq % len(keys)]))

    threads = [threading.Thread(target=submit, args=(batch,)) for batch in batches]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    calls = writer.stop()
    assert [call["seq"] for call in calls] == list(range(200))
    assert calls == [make_record(seq, keys[seq % len(keys)]) for seq in range(200)]
    assert load_session(tmp_path / "run") == calls
    assert codex_probe.load_session(tmp_path / "run") == calls


def test_every_line_is_flushed_while_the_writer_is_still_running(
    make_writer: Callable[..., RunWriter], tmp_path: Path
) -> None:
    writer = make_writer()
    path = tmp_path / "run" / session_file_name("session-A")

    def wait_for_lines(count: int) -> list[dict[str, Any]]:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if path.exists():
                lines = path.read_bytes().splitlines()
                if len(lines) == count:
                    return [json.loads(line) for line in lines]
            time.sleep(0.01)
        raise AssertionError(f"{count} line(s) were not flushed within 5 s")

    writer.submit_call(make_record(0))
    assert wait_for_lines(1) == [make_record(0)]
    writer.submit_call(make_record(1))
    assert wait_for_lines(2) == [make_record(0), make_record(1)]
    for line in path.read_bytes().splitlines():
        validate_call_record(json.loads(line))
    writer.stop()


def test_single_jsonl_file_can_be_loaded(make_writer: Callable[..., RunWriter], tmp_path: Path) -> None:
    writer = make_writer()
    for seq in (3, 1, 2):
        writer.submit_call(make_record(seq, "A"))
    writer.submit_call(make_record(0, "B"))
    writer.stop()
    assert [c["seq"] for c in load_session(tmp_path / "run" / session_file_name("A"))] == [1, 2, 3]


def test_non_llm_requests_never_appear_in_calls(make_writer: Callable[..., RunWriter], tmp_path: Path) -> None:
    writer = make_writer()
    writer.submit_request_log({"kind": "request", "method": "GET", "path": "/v1/models", "status": 200})
    writer.submit_call(make_record(0))
    calls = writer.stop()
    run = tmp_path / "run"
    assert [c["seq"] for c in calls] == [0]
    assert load_session(run) == calls
    other = [json.loads(line) for line in (run / OTHER_REQUESTS_FILE).read_text("utf-8").splitlines()]
    assert other == [{"kind": "request", "method": "GET", "path": "/v1/models", "status": 200}]


def test_metadata_file_is_rewritten_and_not_read_as_calls(
    make_writer: Callable[..., RunWriter], tmp_path: Path
) -> None:
    writer = make_writer()
    writer.write_metadata({"started_at": "2026-10-24T10:00:00Z", "seed": 0})
    writer.submit_call(make_record(0))
    writer.write_metadata({"started_at": "2026-10-24T10:00:00Z", "seed": 0, "finished_at": "2026-10-24T10:05:00Z"})
    calls = writer.stop()
    run = tmp_path / "run"
    assert json.loads((run / RUN_METADATA_FILE).read_text("utf-8"))["finished_at"] == "2026-10-24T10:05:00Z"
    assert not (run / (RUN_METADATA_FILE + ".tmp")).exists()
    assert load_session(run) == calls


def test_empty_run_directory_has_no_calls(tmp_path: Path) -> None:
    assert load_session(tmp_path) == []


# --- validation at submit time -----------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.pop("outcome"), r"record\.outcome: required field is missing"),
        (lambda r: r.update(kind="request"), "record.kind must be 'llm_call'"),
        (lambda r: r.update(seq="1"), r"record\.seq must be a non-negative integer"),
        (lambda r: r.update(seq=-1), r"record\.seq must be a non-negative integer"),
        (lambda r: r.update(call_id=""), r"record\.call_id must be a non-empty string"),
        (lambda r: r.update(session_key=5), r"record\.session_key must be a string or null"),
        (lambda r: r.update(outcome="mostly fine"), r"record\.outcome must be one of"),
        (lambda r: r["upstream"].pop("url"), r"record\.upstream\.url: required field is missing"),
        (lambda r: r["request"].update(headers=[("a", "b")]), r"record\.request\.headers\[0\] must be a \[name, value\] pair"),
        (lambda r: r["request"].update(body={"text": "x", "base64": "eA=="}), "exactly one of 'text' or 'base64'"),
        (lambda r: r["request"].update(body={"base64": "***"}), r"record\.request\.body\.base64 is not valid base64"),
        (lambda r: r["request"].update(body={"base64": "eA==", "json": 1}), "a base64 body cannot have a 'json' view"),
        (lambda r: r["response"].update(status=True), r"record\.response\.status must be a non-negative integer"),
        (lambda r: r["response"].pop("source"), r"record\.response\.source: required field is missing"),
        (lambda r: r["response"].update(source="cache"), r"record\.response\.source must be 'upstream', 'proxy' or null"),
        (lambda r: r["response"].update(source=None), "must be null when source is null"),
        (lambda r: r["response"].update(source="proxy", status=None), r"record\.response\.status must be a non-negative integer"),
        (lambda r: r["response"].update(headers=None), r"record\.response\.headers must be a list"),
        (lambda r: r.update(stream={"events": {}}), r"record\.stream\.events must be a list"),
        (lambda r: r.update(stream={"events": []}), r"record\.stream\.reconstruction: required field is missing"),
        (lambda r: r["timing"].update(total_ms=-1), r"record\.timing\.total_ms must be a non-negative number or null"),
        (lambda r: r.update(derived=None), r"record\.derived must be an object"),
    ],
)
def test_invalid_records_are_rejected(
    make_writer: Callable[..., RunWriter], mutate: Callable[[dict[str, Any]], Any], message: str
) -> None:
    record = make_record(0)
    mutate(record)
    with pytest.raises(RecordError, match=message):
        validate_call_record(record)
    writer = make_writer()
    with pytest.raises(RecordError, match=message):
        writer.submit_call(record)


def test_records_without_a_response_or_stream_are_valid() -> None:
    record = make_record(0, outcome="client_disconnected", stream=None)
    record["response"] = {"source": None, "status": None, "headers": None, "body": None}
    record["timing"].update(first_byte_ms=None, total_ms=None)
    validate_call_record(record)


def test_duplicate_seq_and_submit_after_stop_are_rejected(make_writer: Callable[..., RunWriter]) -> None:
    writer = make_writer()
    writer.submit_call(make_record(0))
    with pytest.raises(RecordError, match="seq 0 was already submitted"):
        writer.submit_call(make_record(0, "other"))
    assert [c["call_id"] for c in writer.stop()] == ["call-0"]
    with pytest.raises(RuntimeError, match="stopped"):
        writer.submit_call(make_record(1))


# --- write failures are reported, never silent -------------------------------


def test_unwritable_run_directory_makes_stop_raise_with_the_calls(
    make_writer: Callable[..., RunWriter], tmp_path: Path
) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a file where the run directory should go")
    writer = make_writer(blocker / "run")
    writer.submit_call(make_record(1))
    writer.submit_call(make_record(0))
    with pytest.raises(LogWriteError) as info:
        writer.stop()
    assert [c["seq"] for c in info.value.calls] == [0, 1]
    assert info.value.calls == [make_record(0), make_record(1)]
    assert len(info.value.errors) == 2
    assert "log write(s) failed" in str(info.value)


def test_one_failed_record_is_reported_and_the_others_are_written(
    make_writer: Callable[..., RunWriter], tmp_path: Path
) -> None:
    writer = make_writer()
    unserializable = make_record(1, derived={"not json": {1, 2}})
    writer.submit_call(make_record(0))
    writer.submit_call(unserializable)
    writer.submit_call(make_record(2))
    with pytest.raises(LogWriteError) as info:
        writer.stop()
    assert [c["seq"] for c in info.value.calls] == [0, 1, 2]
    assert len(info.value.errors) == 1 and info.value.errors[0].startswith("call seq 1: TypeError")
    assert [c["seq"] for c in load_session(tmp_path / "run")] == [0, 2]


# --- load_session refuses damaged files --------------------------------------


@pytest.fixture
def written_run(make_writer: Callable[..., RunWriter], tmp_path: Path) -> Path:
    """A run directory holding two valid records of session A."""
    writer = make_writer()
    writer.submit_call(make_record(0))
    writer.submit_call(make_record(1))
    writer.stop()
    return tmp_path / "run"


def _session_a(run: Path) -> Path:
    return run / session_file_name("session-A")


def _replace_line(run: Path, line_number: int, text: str) -> None:
    path = _session_a(run)
    lines = path.read_text("utf-8").splitlines()
    lines[line_number - 1] = text
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record_with(change: Callable[[dict[str, Any]], Any]) -> str:
    record = copy.deepcopy(make_record(1))
    change(record)
    return json.dumps(record)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda run: _replace_line(run, 2, '{"kind": "llm_call", "seq": '), r":2: not valid JSON"),
        (lambda run: _replace_line(run, 1, "[1, 2, 3]"), r":1: line is not a JSON object"),
        (lambda run: _replace_line(run, 2, _record_with(lambda r: r.pop("timing"))), r":2: record\.timing: required field is missing"),
        (lambda run: _replace_line(run, 2, _record_with(lambda r: r.update(seq="1"))), r":2: record\.seq must be a non-negative integer"),
        (lambda run: _replace_line(run, 1, "   "), r":1: blank line"),
        (lambda run: _replace_line(run, 2, _record_with(lambda r: r.update(seq=0))), r":2: seq 0 also appears at .*:1"),
        (lambda run: _session_a(run).write_bytes(_session_a(run).read_bytes()[:-30]), r":2: last line has no newline"),
        (lambda run: _session_a(run).write_bytes(b""), r": file is empty"),
        (lambda run: (run / NO_SESSION_FILE).write_bytes(b""), r"no-session\.jsonl: file is empty"),
        (lambda run: _replace_line(run, 1, _record_with(lambda r: r.update(session_key="session-B"))), r":1: this record belongs in session-"),
    ],
)
def test_damaged_files_fail_with_file_and_line(
    written_run: Path, damage: Callable[[Path], Any], message: str
) -> None:
    damage(written_run)
    with pytest.raises(SessionLoadError, match=message):
        load_session(written_run)


def test_missing_path_fails(tmp_path: Path) -> None:
    with pytest.raises(SessionLoadError, match="no such run directory or JSONL file"):
        load_session(tmp_path / "nowhere")
