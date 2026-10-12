#!/usr/bin/env python3
"""Standalone server for people who clone only the ai-origin repository.

Only the dashboard asset directory is public. Python, prompts and experiment
records are outside that directory. The Kvasir dev server mounts the same assets
and HTTP adapter, but this entry point has no dependency on Kvasir.
"""
from __future__ import annotations

import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

FRONTEND_ROOT = Path(__file__).resolve().parents[1] / "html" / "ai-origin"


class DevHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Serve the same /ai-origin/ URLs as the integrated Kvasir server."""

    def _api(self) -> None:
        # Static pages remain accessible even with an unprepared interpreter.
        try:
            from http_api import handle
        except ImportError:
            self.send_error(503, "Install requirements.txt using this Python environment first")
            return
        handle(self)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path.startswith("/ai-origin/api/"):
            self._api()
        elif path == "/":
            self.send_response(302)
            self.send_header("Location", "/ai-origin/")
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            super().do_GET()

    def do_HEAD(self) -> None:
        if urlsplit(self.path).path.startswith("/ai-origin/api/"):
            self.send_error(405, "Use GET for the experiment API")
        else:
            super().do_HEAD()

    def do_POST(self) -> None:
        if urlsplit(self.path).path.startswith("/ai-origin/api/"):
            self._api()
        else:
            self.send_error(405, "Method not allowed")

    def translate_path(self, path: str) -> str:
        clean = unquote(urlsplit(path).path)
        root = FRONTEND_ROOT.resolve()
        if clean == "/ai-origin" or clean.startswith("/ai-origin/"):
            candidate = (root / clean.removeprefix("/ai-origin").lstrip("/")).resolve()
            # Resolve symlinks and encoded '..' before checking containment.
            if candidate.is_relative_to(root):
                return str(candidate)
        return str(root / ".not-found")


def main() -> None:
    port = int(os.environ.get("PORT", "8383"))
    host = os.environ.get("HOST", "127.0.0.1")
    with ThreadingHTTPServer((host, port), DevHTTPRequestHandler) as server:
        print(f"AI-origin lab: http://{host}:{port}/ai-origin/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
