"""Readable file storage with atomic replacement and a single-writer lock.

Each record gets its own JSON file. Writing to a temporary sibling and renaming
it means dashboard readers see either the old or new complete JSON document.
The process-level lock prevents a second dev server from interrupting this
server's active jobs; the RLock protects read-modify-write operations in threads.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = threading.RLock()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._process_lock = (root / ".server.lock").open("a")
        try:
            fcntl.flock(self._process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._process_lock.close()
            raise RuntimeError("Another AI-origin server already uses this data directory.") from None
        for collection in ("samples", "runs", "improvements", "candidates"):
            (root / collection).mkdir(exist_ok=True, mode=0o700)
        for collection in ("runs", "improvements"):
            # We cannot know whether an interrupted HTTP call was already
            # charged. Keep partial results and require an explicit new run.
            for item in self.list(collection):
                if item["status"] in {"queued", "running", "cancelling"}:
                    item.update(status="interrupted", finished_at=now(), error="Server restarted. Saved results retained; no automatic paid retries.")
                    self.write(collection, item)

    def path(self, collection: str, item_id: str, suffix: str = ".json") -> Path:
        """IDs are server-generated, never arbitrary user-supplied file paths."""
        if collection not in {"samples", "runs", "improvements", "candidates"}:
            raise ValueError("Unknown collection")
        if not re.fullmatch(r"[a-z]+_[a-f0-9]{16}", item_id):
            raise ValueError("Invalid record ID")
        return self.root / collection / (item_id + suffix)

    def atomic(self, path: Path, content: str) -> None:
        """Same-directory rename is atomic on the local filesystem."""
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def write(self, collection: str, item: dict[str, Any]) -> None:
        with self.lock:
            self.atomic(self.path(collection, item["id"]), json.dumps(item, ensure_ascii=False, indent=2, allow_nan=False))

    def text(self, collection: str, item_id: str, content: str) -> None:
        with self.lock:
            self.atomic(self.path(collection, item_id, ".txt"), content)

    def get(self, collection: str, item_id: str) -> dict[str, Any]:
        with self.lock:
            return json.loads(self.path(collection, item_id).read_text(encoding="utf-8"))

    def delete(self, collection: str, item_id: str) -> None:
        with self.lock:
            path = self.path(collection, item_id)
            if not path.exists():
                raise FileNotFoundError(item_id)
            path.unlink()
            text = path.with_suffix(".txt")
            if text.exists():
                text.unlink()

    def list(self, collection: str) -> list[dict[str, Any]]:
        with self.lock:
            items = [json.loads(path.read_text(encoding="utf-8")) for path in (self.root / collection).glob("*.json")]
            return sorted(items, key=lambda item: item["created_at"], reverse=True)

    def close(self) -> None:
        self._process_lock.close()
