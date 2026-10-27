"""A small local web server for the viewer, using only the standard library.

It serves one page and two JSON endpoints, and binds to the loopback address by default. Two
cheap checks keep other web pages from using it as a back door: requests must carry a
``localhost`` Host header (so a DNS-rebinding page is refused), and the POST endpoint only
accepts ``application/json`` (which a cross-site form cannot send).
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from foilnet.viewer.service import ViewerError, ViewerModel

MAX_BODY_BYTES = 4096
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}
CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "connect-src 'self'; img-src data:"
)


def _page() -> bytes:
    return resources.files("foilnet.viewer").joinpath("static/index.html").read_bytes()


def _host_is_local(header: str | None) -> bool:
    if not header:
        return False
    host = header.rsplit(":", 1)[0] if not header.endswith("]") and ":" in header else header
    return host.lower() in LOCAL_HOSTS


def make_handler(model: ViewerModel) -> type[BaseHTTPRequestHandler]:
    page = _page()
    info = json.dumps(model.info()).encode()

    class Handler(BaseHTTPRequestHandler):
        server_version = "FoilNetViewer"

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass  # one line per slider movement is noise

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: object) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _local_only(self) -> bool:
            if _host_is_local(self.headers.get("Host")):
                return True
            self._json(HTTPStatus.FORBIDDEN, {"error": "only local access is allowed"})
            return False

        def do_GET(self) -> None:  # noqa: N802
            if not self._local_only():
                return
            if self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._send(HTTPStatus.OK, info, "application/json")
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._local_only():
                return
            if self.path != "/api/evaluate":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if self.headers.get_content_type() != "application/json":
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "send application/json"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = -1
            if not 0 < length <= MAX_BODY_BYTES:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "missing or oversized request body"})
                return
            try:
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ViewerError("the request must be a JSON object")
                sweep = request.get("sweep", True) is not False
                result = model.evaluate(request, sweep=sweep)
            except json.JSONDecodeError:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "the request is not valid JSON"})
            except ViewerError as exc:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})
            else:
                self._json(HTTPStatus.OK, result)

    return Handler


def make_server(
    model: ViewerModel, host: str = "127.0.0.1", port: int = 8000
) -> ThreadingHTTPServer:
    """A server ready to ``serve_forever()``; ``port=0`` picks a free port."""
    return ThreadingHTTPServer((host, port), make_handler(model))
