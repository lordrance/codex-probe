"""Load and validate a CodexProbe config.

A config is a JSON-compatible dict, or the path to a JSON file holding one::

    {
        "upstream": {
            "name": "openai",
            "base_url": "https://api.openai.com/v1",
            "wire_api": "responses",
            "auth": {"mode": "passthrough"}
        },
        "listen": {"host": "127.0.0.1", "port": 0},
        "log_dir": "runs",
        "seed": 0
    }

Only ``upstream.base_url`` and ``log_dir`` are required. Validation is strict:
unknown keys are rejected at every level, so a typo cannot silently change
behaviour. Secrets are never accepted in the config itself; ``auth.mode = "env"``
names the environment variable that holds the key instead.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

AuthMode = Literal["passthrough", "none", "env"]

AUTH_MODES: tuple[str, ...] = ("passthrough", "none", "env")
LISTEN_HOST = "127.0.0.1"

_ROOT_KEYS = frozenset({"upstream", "listen", "log_dir", "seed"})
_UPSTREAM_KEYS = frozenset({"name", "base_url", "wire_api", "auth"})
_AUTH_KEYS = frozenset({"mode", "env_var"})
_LISTEN_KEYS = frozenset({"host", "port"})
_ENV_VAR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ConfigError(ValueError):
    """Raised when a config is missing, unreadable or invalid; the message names the field."""


@dataclass(frozen=True)
class AuthConfig:
    """How the proxy authenticates to the upstream backend.

    ``passthrough`` forwards the credentials Codex sent, ``none`` forwards no
    credentials (for local backends), and ``env`` makes the proxy send the key
    stored in the environment variable named by ``env_var``.
    """

    mode: AuthMode = "passthrough"
    env_var: str | None = None


@dataclass(frozen=True)
class UpstreamConfig:
    """The real backend that every request is forwarded to."""

    base_url: str
    name: str
    wire_api: Literal["responses"] = "responses"
    auth: AuthConfig = field(default_factory=AuthConfig)


@dataclass(frozen=True)
class ListenConfig:
    """Where the proxy listens: loopback only; port 0 picks a free port."""

    host: str = LISTEN_HOST
    port: int = 0


@dataclass(frozen=True)
class ProbeConfig:
    """A validated CodexProbe config."""

    upstream: UpstreamConfig
    log_dir: Path
    listen: ListenConfig = field(default_factory=ListenConfig)
    seed: int = 0


def load_config(source: Mapping[str, Any] | str | os.PathLike[str]) -> ProbeConfig:
    """Validate a config given as a dict, or as the path to a JSON file.

    Raises:
        ConfigError: the file cannot be read or parsed, or a field is missing,
            unknown or invalid.
    """
    if isinstance(source, Mapping):
        raw: Any = source
    elif isinstance(source, (str, os.PathLike)):
        raw = _read_json_file(Path(source))
    else:
        raise ConfigError(
            f"config must be a dict or the path to a JSON file, got {type(source).__name__}"
        )
    return _parse_root(raw)


def _read_json_file(path: Path) -> Any:
    try:
        # utf-8-sig also accepts the BOM that some Windows editors add.
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from None
    except UnicodeDecodeError:
        raise ConfigError(f"config file {path} is not valid UTF-8") from None
    if not text.strip():
        raise ConfigError(f"config file {path} is empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file {path} is not valid JSON: {exc}") from None


def _parse_root(raw: Any) -> ProbeConfig:
    obj = _expect_object(raw, "config")
    _reject_unknown(obj, _ROOT_KEYS, "config")
    upstream = _parse_upstream(_require(obj, "upstream", "upstream"))
    log_dir = Path(_expect_str(_require(obj, "log_dir", "log_dir"), "log_dir"))
    listen = _parse_listen(obj.get("listen", {}))
    seed = _expect_int(obj.get("seed", 0), "seed")
    return ProbeConfig(upstream=upstream, log_dir=log_dir, listen=listen, seed=seed)


def _parse_upstream(raw: Any) -> UpstreamConfig:
    obj = _expect_object(raw, "upstream")
    _reject_unknown(obj, _UPSTREAM_KEYS, "upstream")
    base_url = _parse_base_url(_require(obj, "base_url", "upstream.base_url"))
    if "name" in obj:
        name = _expect_str(obj["name"], "upstream.name")
    else:
        name = urlsplit(base_url).netloc
    _parse_wire_api(obj.get("wire_api", "responses"))
    auth = _parse_auth(obj["auth"]) if "auth" in obj else AuthConfig()
    return UpstreamConfig(base_url=base_url, name=name, wire_api="responses", auth=auth)


def _parse_base_url(value: Any) -> str:
    url = _expect_str(value, "upstream.base_url")
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise ConfigError(f"upstream.base_url is not a valid URL: {exc}") from None
    # Checked first so that no later message echoes a URL that holds a password.
    if parts.username is not None or parts.password is not None:
        raise ConfigError(
            "upstream.base_url must not contain credentials (user:password@); "
            "configure upstream.auth instead"
        )
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ConfigError(
            f"upstream.base_url must be an absolute http:// or https:// URL, got {url!r}"
        )
    try:
        parts.port
    except ValueError:
        raise ConfigError(f"upstream.base_url has an invalid port: {url!r}") from None
    if not parts.hostname:
        raise ConfigError(f"upstream.base_url has no host: {url!r}")
    if parts.query or parts.fragment:
        raise ConfigError(
            f"upstream.base_url must not contain a query string or fragment: {url!r}"
        )
    # Request paths are appended as "/<rest>", so a trailing slash would double up.
    return url.rstrip("/")


def _parse_wire_api(value: Any) -> None:
    wire_api = _expect_str(value, "upstream.wire_api")
    if wire_api == "responses":
        return
    if wire_api == "chat":
        raise ConfigError(
            "upstream.wire_api: 'chat' (Chat Completions) is not supported. Codex 0.164.0 "
            "only speaks the Responses API (Chat Completions support was removed in Codex "
            "0.95.0), and CodexProbe forwards traffic unchanged: it does not translate "
            "between protocols. Use wire_api 'responses' with a backend that serves "
            "/v1/responses, such as Ollama."
        )
    raise ConfigError(f"upstream.wire_api must be 'responses', got {wire_api!r}")


def _parse_auth(raw: Any) -> AuthConfig:
    obj = _expect_object(raw, "upstream.auth")
    _reject_unknown(obj, _AUTH_KEYS, "upstream.auth")
    mode = _expect_str(_require(obj, "mode", "upstream.auth.mode"), "upstream.auth.mode")
    if mode not in AUTH_MODES:
        raise ConfigError(
            f"upstream.auth.mode must be one of {', '.join(AUTH_MODES)}; got {mode!r}"
        )
    if mode != "env":
        if "env_var" in obj:
            raise ConfigError(
                f"upstream.auth.env_var is only allowed when mode is 'env' (mode is {mode!r})"
            )
        return AuthConfig(mode=cast(AuthMode, mode))
    if "env_var" not in obj:
        raise ConfigError(
            "upstream.auth.env_var: required when mode is 'env' "
            "(the name of the environment variable that holds the key)"
        )
    env_var = _expect_str(obj["env_var"], "upstream.auth.env_var")
    # The value is not echoed: if someone pasted a key here, it must not reach a log.
    if not _ENV_VAR_NAME.fullmatch(env_var):
        raise ConfigError(
            "upstream.auth.env_var must be an environment variable name such as "
            "OPENAI_API_KEY (letters, digits and underscores). Put the key itself in that "
            "variable, never in the config."
        )
    return AuthConfig(mode="env", env_var=env_var)


def _parse_listen(raw: Any) -> ListenConfig:
    obj = _expect_object(raw, "listen")
    _reject_unknown(obj, _LISTEN_KEYS, "listen")
    host = _expect_str(obj.get("host", LISTEN_HOST), "listen.host")
    if host != LISTEN_HOST:
        raise ConfigError(
            f"listen.host must be {LISTEN_HOST!r} (CodexProbe only listens on loopback), "
            f"got {host!r}"
        )
    port = _expect_int(obj.get("port", 0), "listen.port")
    if not 0 <= port <= 65535:
        raise ConfigError(
            f"listen.port must be between 0 and 65535 (0 picks a free port), got {port}"
        )
    return ListenConfig(host=host, port=port)


def _require(obj: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in obj:
        raise ConfigError(f"{path}: required field is missing")
    return obj[key]


def _reject_unknown(obj: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(str(key) for key in obj if key not in allowed)
    if unknown:
        raise ConfigError(
            f"{path}: unknown field(s) {', '.join(unknown)}; "
            f"allowed: {', '.join(sorted(allowed))}"
        )


def _expect_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a JSON object, got {_json_type(value)}")
    return value


def _expect_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path} must be a non-empty string, got {_json_type(value)}")
    if not value.strip():
        raise ConfigError(f"{path} must be a non-empty string, got an empty string")
    return value


def _expect_int(value: Any, path: str) -> int:
    # bool is a subclass of int in Python, but true/false is never a valid number here.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer, got {_json_type(value)}")
    return value


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    return type(value).__name__
