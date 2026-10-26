"""Run an ASGI app with uvicorn on a background thread, bound to 127.0.0.1.

The proxy and the mock server both use this. The socket is bound here, before
uvicorn starts, so port 0 works and a busy port fails with a clear error.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any, Literal

import uvicorn

HOST = "127.0.0.1"


class ServerStartError(RuntimeError):
    """The server could not start."""


class AbortConnection(Exception):
    """Raise inside an ASGI app after the response has started to make uvicorn drop the connection.

    The client then sees the body end abruptly, which is how a cut upstream
    stream is passed on without being disguised as a complete one.
    """


class _HideAbortTraceback(logging.Filter):
    """Stop uvicorn logging a traceback for a deliberate :class:`AbortConnection`."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.exc_info and isinstance(record.exc_info[1], AbortConnection))


_ABORT_FILTER = _HideAbortTraceback()


class BackgroundServer:
    """One uvicorn server on a daemon thread, with blocking start() and stop()."""

    def __init__(
        self,
        app: Any,
        *,
        port: int = 0,
        lifespan: Literal["on", "off"] = "off",
        graceful_timeout: int = 2,
        thread_name: str = "codexprobe-server",
    ) -> None:
        self._app = app
        self._requested_port = port
        self._lifespan = lifespan
        self._graceful_timeout = graceful_timeout
        self._thread_name = thread_name
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.port: int | None = None
        """The bound port while running, else None."""

    def start(self, timeout: float = 10.0) -> int:
        """Bind 127.0.0.1, start serving, and return the port.

        Raises:
            ServerStartError: already running, the port cannot be bound, or the
                server did not come up within ``timeout`` seconds.
        """
        if self._thread is not None:
            raise ServerStartError("the server is already running")
        logger = logging.getLogger("uvicorn.error")
        if _ABORT_FILTER not in logger.filters:
            logger.addFilter(_ABORT_FILTER)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((HOST, self._requested_port))
        except OSError as exc:
            sock.close()
            raise ServerStartError(f"cannot listen on {HOST}:{self._requested_port}: {exc}") from exc
        port = sock.getsockname()[1]
        config = uvicorn.Config(
            self._app,
            interface="asgi3",  # a bound method is not detected as ASGI 3 automatically
            lifespan=self._lifespan,
            log_config=None,
            log_level="warning",
            access_log=False,
            proxy_headers=False,
            server_header=False,  # never add a Server header of our own
            date_header=False,  # never add a Date header of our own
            timeout_graceful_shutdown=self._graceful_timeout,
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [sock]}, name=self._thread_name, daemon=True
        )
        thread.start()
        deadline = time.monotonic() + timeout
        while not server.started:
            if not thread.is_alive():
                sock.close()
                raise ServerStartError("the server exited during startup")
            if time.monotonic() > deadline:
                server.should_exit = True
                thread.join(5)
                sock.close()
                raise ServerStartError(f"the server did not start within {timeout} s")
            time.sleep(0.01)
        self._server, self._thread, self.port = server, thread, port
        return port

    def stop(self, timeout: float = 10.0) -> None:
        """Stop serving and release the port. Calling it again, or before start(), does nothing.

        Requests still running get the graceful-shutdown period, then are
        cancelled. If the thread is still alive after ``timeout`` seconds,
        uvicorn is forced to exit.
        """
        server, thread = self._server, self._thread
        if server is None or thread is None:
            return
        server.should_exit = True
        thread.join(timeout)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout)
        self._server = self._thread = None
        self.port = None
        if thread.is_alive():
            raise RuntimeError(f"the {self._thread_name} thread did not stop")
