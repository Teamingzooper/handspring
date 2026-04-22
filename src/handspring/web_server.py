"""Tiny MJPEG HTTP server that streams the annotated camera feed.

Used by Plash (or any browser) to show handspring's camera output as a
desktop background. The main loop writes the latest JPEG-encoded frame into
a shared `LatestFrame` buffer; connected clients stream it as
`multipart/x-mixed-replace`.
"""

from __future__ import annotations

import http.server
import socketserver
import threading
from pathlib import Path

_INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>handspring</title>
  <style>
    html, body { margin: 0; padding: 0; height: 100%; background: #000; overflow: hidden; }
    img { position: fixed; inset: 0; width: 100vw; height: 100vh; object-fit: cover;
          -webkit-user-select: none; user-select: none; pointer-events: none; }
  </style>
</head>
<body>
  <img src="/stream" alt="handspring stream">
</body>
</html>
"""


class LatestFrame:
    """Thread-safe holder for the most recent JPEG-encoded frame."""

    def __init__(self) -> None:
        self._bytes: bytes | None = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def set(self, jpeg: bytes) -> None:
        with self._cond:
            self._bytes = jpeg
            self._cond.notify_all()

    def wait_next(self, timeout: float = 1.0) -> bytes | None:
        with self._cond:
            self._cond.wait(timeout=timeout)
            return self._bytes


def _make_handler(latest: LatestFrame) -> type[http.server.BaseHTTPRequestHandler]:
    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:  # quiet
            pass

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                body = _INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                self.end_headers()
                try:
                    while True:
                        frame = latest.wait_next(timeout=1.0)
                        if frame is None:
                            continue
                        try:
                            self.wfile.write(b"--FRAME\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                        except (BrokenPipeError, ConnectionResetError):
                            return
                except Exception:  # noqa: BLE001
                    return
            else:
                self.send_error(404, "not found")

    return _Handler


class _ReusableThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class WebServer:
    def __init__(self, port: int, latest: LatestFrame) -> None:
        self._port = port
        self._latest = latest
        self._server: _ReusableThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = _make_handler(self._latest)
        self._server = _ReusableThreadingHTTPServer(("127.0.0.1", self._port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def write_static_site(root: Path) -> None:
    """Write an index.html into `root` that points at http://localhost:8765/stream.

    Use this for Plash: point Plash at this folder (or just at http://localhost:8765).
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text(_INDEX_HTML, encoding="utf-8")


# Expose the HTML for tests / users who want to write it themselves.
INDEX_HTML = _INDEX_HTML
