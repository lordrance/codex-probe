"""Tests for codex_probe.config: loading from a dict or a JSON file, defaults, and rejections."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from codex_probe.config import (
    AuthConfig,
    ConfigError,
    ListenConfig,
    ProbeConfig,
    UpstreamConfig,
    load_config,
)

Config = dict[str, Any]


def minimal() -> Config:
    """The smallest valid config: only the two required fields."""
    return {"upstream": {"base_url": "https://api.openai.com/v1"}, "log_dir": "runs"}


def full() -> Config:
    """A config that sets every field explicitly."""
    return {
        "upstream": {
            "name": "openai",
            "base_url": "https://api.openai.com/v1",
            "wire_api": "responses",
            "auth": {"mode": "env", "env_var": "OPENAI_API_KEY"},
        },
        "listen": {"host": "127.0.0.1", "port": 8135},
        "log_dir": "D:/codexprobe/runs",
        "seed": 7,
    }


# --- valid configs ---------------------------------------------------------


def test_minimal_config_gets_documented_defaults() -> None:
    cfg = load_config(minimal())
    assert cfg.upstream.base_url == "https://api.openai.com/v1"
    assert cfg.upstream.name == "api.openai.com"
    assert cfg.upstream.wire_api == "responses"
    assert cfg.upstream.auth == AuthConfig(mode="passthrough", env_var=None)
    assert cfg.listen == ListenConfig(host="127.0.0.1", port=0)
    assert cfg.seed == 0
    assert cfg.log_dir == Path("runs")


def test_full_config_is_loaded_field_by_field() -> None:
    assert load_config(full()) == ProbeConfig(
        upstream=UpstreamConfig(
            base_url="https://api.openai.com/v1",
            name="openai",
            wire_api="responses",
            auth=AuthConfig(mode="env", env_var="OPENAI_API_KEY"),
        ),
        log_dir=Path("D:/codexprobe/runs"),
        listen=ListenConfig(host="127.0.0.1", port=8135),
        seed=7,
    )


def test_local_http_backend_with_auth_none() -> None:
    cfg = load_config(
        {
            "upstream": {"base_url": "http://127.0.0.1:11434/v1/", "auth": {"mode": "none"}},
            "log_dir": "runs",
        }
    )
    assert cfg.upstream.base_url == "http://127.0.0.1:11434/v1"  # trailing slash removed
    assert cfg.upstream.name == "127.0.0.1:11434"
    assert cfg.upstream.auth == AuthConfig(mode="none")


def test_partial_listen_keeps_default_host() -> None:
    cfg = load_config({**minimal(), "listen": {"port": 9000}})
    assert cfg.listen == ListenConfig(host="127.0.0.1", port=9000)


def test_negative_seed_is_allowed() -> None:
    assert load_config({**minimal(), "seed": -5}).seed == -5


def test_input_dict_is_not_modified() -> None:
    cfg = full()
    snapshot = copy.deepcopy(cfg)
    load_config(cfg)
    assert cfg == snapshot


# --- JSON files --------------------------------------------------------------


def test_load_from_json_file_as_str_and_path(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(full()), encoding="utf-8")
    assert load_config(str(path)) == load_config(full())
    assert load_config(path) == load_config(full())


def test_json_file_with_utf8_bom_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(minimal()).encode("utf-8"))
    assert load_config(path).upstream.name == "api.openai.com"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "is empty"),
        (b"  \n\t ", "is empty"),
        (b'{"upstream": {"base_url": ', "is not valid JSON"),
        (b"{'single': 'quotes'}", "is not valid JSON"),
        (b"[1, 2, 3]", "config must be a JSON object, got array"),
        (b'"just a string"', "config must be a JSON object, got string"),
        (b"null", "config must be a JSON object, got null"),
        (b"\xff\xfe\x00not utf-8", "is not valid UTF-8"),
    ],
)
def test_bad_json_files_are_rejected(tmp_path: Path, content: bytes, message: str) -> None:
    path = tmp_path / "config.json"
    path.write_bytes(content)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "does-not-exist.json")


def test_unsupported_source_type_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must be a dict or the path to a JSON file"):
        load_config(42)  # type: ignore[arg-type]


def test_config_error_is_a_value_error() -> None:
    assert issubclass(ConfigError, ValueError)


# --- rejected fields ---------------------------------------------------------


def _set(path: str, value: Any) -> Callable[[Config], None]:
    """Return a mutator that sets a dotted path, e.g. 'upstream.auth.mode'."""
    *parents, leaf = path.split(".")

    def mutate(cfg: Config) -> None:
        node = cfg
        for key in parents:
            node = node[key]
        node[leaf] = value

    return mutate


def _delete(path: str) -> Callable[[Config], None]:
    """Return a mutator that deletes a dotted path."""
    *parents, leaf = path.split(".")

    def mutate(cfg: Config) -> None:
        node = cfg
        for key in parents:
            node = node[key]
        del node[leaf]

    return mutate


REJECTED: list[tuple[str, Callable[[Config], None], str]] = [
    # unknown fields, at every level
    ("unknown root field", _set("extra", 1), r"config: unknown field\(s\) extra"),
    ("unknown upstream field", _set("upstream.model", "gpt"), r"upstream: unknown field\(s\) model"),
    ("key inside auth", _set("upstream.auth.api_key", "sk-x"), r"upstream\.auth: unknown field\(s\) api_key"),
    ("unknown listen field", _set("listen.address", "x"), r"listen: unknown field\(s\) address"),
    # missing required fields and wrong shapes
    ("missing upstream", _delete("upstream"), "upstream: required field is missing"),
    ("missing base_url", _delete("upstream.base_url"), r"upstream\.base_url: required field is missing"),
    ("missing log_dir", _delete("log_dir"), "log_dir: required field is missing"),
    ("upstream is a list", _set("upstream", []), "upstream must be a JSON object, got array"),
    ("listen is null", _set("listen", None), "listen must be a JSON object, got null"),
    # strings
    ("base_url is a number", _set("upstream.base_url", 5), r"upstream\.base_url must be a non-empty string, got number"),
    ("base_url is empty", _set("upstream.base_url", ""), r"upstream\.base_url must be a non-empty string"),
    ("base_url is blank", _set("upstream.base_url", "   "), r"upstream\.base_url must be a non-empty string"),
    ("name is empty", _set("upstream.name", ""), r"upstream\.name must be a non-empty string"),
    ("log_dir is empty", _set("log_dir", ""), "log_dir must be a non-empty string"),
    ("log_dir is a number", _set("log_dir", 5), "log_dir must be a non-empty string, got number"),
    # upstream URL
    ("ftp scheme", _set("upstream.base_url", "ftp://example.com/v1"), "absolute http:// or https:// URL"),
    ("relative URL", _set("upstream.base_url", "/v1"), "absolute http:// or https:// URL"),
    ("no netloc", _set("upstream.base_url", "http:///v1"), "absolute http:// or https:// URL"),
    ("port but no host", _set("upstream.base_url", "http://:8080/v1"), "has no host"),
    ("port out of range", _set("upstream.base_url", "http://127.0.0.1:99999/v1"), "invalid port"),
    ("query string", _set("upstream.base_url", "https://api.openai.com/v1?x=1"), "query string or fragment"),
    ("fragment", _set("upstream.base_url", "https://api.openai.com/v1#top"), "query string or fragment"),
    # wire_api
    ("wire_api is a number", _set("upstream.wire_api", 1), r"upstream\.wire_api must be a non-empty string"),
    ("unknown wire_api", _set("upstream.wire_api", "completions"), "must be 'responses'"),
    # auth
    ("auth is a string", _set("upstream.auth", "passthrough"), r"upstream\.auth must be a JSON object, got string"),
    ("auth without mode", _set("upstream.auth", {}), r"upstream\.auth\.mode: required field is missing"),
    ("unknown auth mode", _set("upstream.auth.mode", "bearer"), "must be one of passthrough, none, env"),
    ("auth mode is a number", _set("upstream.auth.mode", 1), r"upstream\.auth\.mode must be a non-empty string"),
    ("env without env_var", _delete("upstream.auth.env_var"), "required when mode is 'env'"),
    ("env_var is empty", _set("upstream.auth.env_var", ""), r"upstream\.auth\.env_var must be a non-empty string"),
    ("env_var is a number", _set("upstream.auth.env_var", 1), r"upstream\.auth\.env_var must be a non-empty string"),
    ("env_var on passthrough", _set("upstream.auth", {"mode": "passthrough", "env_var": "OPENAI_API_KEY"}), "only allowed when mode is 'env'"),
    ("env_var on none", _set("upstream.auth", {"mode": "none", "env_var": "OPENAI_API_KEY"}), "only allowed when mode is 'env'"),
    # listen
    ("wildcard host", _set("listen.host", "0.0.0.0"), "listen.host must be '127.0.0.1'"),
    ("localhost host", _set("listen.host", "localhost"), "listen.host must be '127.0.0.1'"),
    ("IPv6 loopback host", _set("listen.host", "::1"), "listen.host must be '127.0.0.1'"),
    ("port is bool", _set("listen.port", True), "listen.port must be an integer, got boolean"),
    ("port is a string", _set("listen.port", "8080"), "listen.port must be an integer, got string"),
    ("port is a float", _set("listen.port", 80.0), "listen.port must be an integer, got number"),
    ("port is negative", _set("listen.port", -1), "between 0 and 65535"),
    ("port too large", _set("listen.port", 65536), "between 0 and 65535"),
    # seed
    ("seed is bool", _set("seed", False), "seed must be an integer, got boolean"),
    ("seed is a float", _set("seed", 1.0), "seed must be an integer, got number"),
    ("seed is a string", _set("seed", "0"), "seed must be an integer, got string"),
]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [pytest.param(mutate, message, id=case) for case, mutate, message in REJECTED],
)
def test_invalid_field_is_rejected_with_clear_message(
    mutate: Callable[[Config], None], message: str
) -> None:
    cfg = full()
    mutate(cfg)
    with pytest.raises(ConfigError, match=message):
        load_config(cfg)


def test_chat_wire_api_explains_why_it_is_unsupported() -> None:
    cfg = full()
    cfg["upstream"]["wire_api"] = "chat"
    with pytest.raises(ConfigError) as info:
        load_config(cfg)
    text = str(info.value)
    assert "Codex 0.164.0" in text
    assert "Chat Completions" in text
    assert "does not translate between protocols" in text


def test_url_with_credentials_is_rejected_without_echoing_them() -> None:
    cfg = full()
    cfg["upstream"]["base_url"] = "https://user:hunter2@api.openai.com/v1"
    with pytest.raises(ConfigError, match="must not contain credentials") as info:
        load_config(cfg)
    assert "hunter2" not in str(info.value)


def test_secret_in_env_var_field_is_rejected_without_echoing_it() -> None:
    cfg = full()
    cfg["upstream"]["auth"]["env_var"] = "sk-proj-abc123"
    with pytest.raises(ConfigError, match="must be an environment variable name") as info:
        load_config(cfg)
    assert "sk-proj-abc123" not in str(info.value)
