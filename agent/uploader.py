"""Buffers signed batches locally and pushes them to the ledger, so an offline
agent never loses records (IMPLEMENTATION PLAN.md §6.2: "buffered locally
when offline"). The finale demo uses InMemoryUploader directly against the
ledger's in-process verify functions; HttpUploader is the real transport.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class Uploader(Protocol):
    def upload_batch(self, device_id: str, batch: list[dict[str, Any]]) -> bool: ...


class InMemoryUploader:
    """For tests and the replay demo: calls a verify callback directly."""

    def __init__(self, on_batch):
        self.on_batch = on_batch
        self.sent: list[tuple[str, list[dict[str, Any]]]] = []

    def upload_batch(self, device_id: str, batch: list[dict[str, Any]]) -> bool:
        self.sent.append((device_id, batch))
        return bool(self.on_batch(device_id, batch))


class FileBufferUploader:
    """Buffers to disk when the ledger is unreachable, and flushes on retry."""

    def __init__(self, buffer_dir: str | Path, inner: Uploader):
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.inner = inner

    def upload_batch(self, device_id: str, batch: list[dict[str, Any]]) -> bool:
        try:
            ok = self.inner.upload_batch(device_id, batch)
        except Exception:
            ok = False
        if not ok:
            self._buffer(device_id, batch)
        return ok

    def _buffer(self, device_id: str, batch: list[dict[str, Any]]) -> None:
        first_seq = batch[0].get("seq", "x") if batch else "x"
        path = self.buffer_dir / f"{device_id.replace(':', '_')}_{first_seq}.json"
        path.write_text(json.dumps(batch))

    def flush(self) -> int:
        """Retries every buffered batch; returns the count that succeeded."""
        sent = 0
        for path in sorted(self.buffer_dir.glob("*.json")):
            batch = json.loads(path.read_text())
            device_id = path.stem.rsplit("_", 1)[0].replace("_", ":", 1)
            if self.inner.upload_batch(device_id, batch):
                path.unlink()
                sent += 1
        return sent
