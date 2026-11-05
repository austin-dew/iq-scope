"""Receiving recordings from the browser.

A file arrives in pieces, and each piece's SHA-256 is checked as it lands. Browsers can't
hash a file in steps, so a single whole-file hash would mean holding all of it in memory;
checking every piece covers every byte just the same, and lets a dropped connection carry
on from the last good piece instead of starting again.

The data goes to a `.part` file, which the library's scan ignores. Only a complete, checked
file that reads as a WAV is renamed into place, so an interrupted upload can't leave
anything that looks finished. What's been received is recorded on this machine's cache, and
that record is the truth: anything in the `.part` past it is cut off before the next write.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from .wavfile import WavError, read_header

PART_BYTES = 32 * 2**20
STALE_AFTER_S = 14 * 24 * 3600
_FREE_MARGIN_BYTES = 2**30
_MIN_WAV_BYTES = 44
_MAX_NAME = 120
_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_WAV_MAGIC = (b"RIFF", b"RF64", b"BW64")


class UploadError(Exception):
    """Something the person uploading should read. `retry` says whether trying again can help."""

    def __init__(self, status: int, message: str, retry: bool = False, **extra):
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry = retry
        self.extra = extra


def safe_name(raw: str) -> str:
    """A plain filename from whatever the browser sent. Any folders in it are thrown away."""
    base = raw.replace("\\", "/").rsplit("/", 1)[-1]
    if not base.lower().endswith(".wav"):
        raise UploadError(415, "Only .wav files can be uploaded.")
    stem = base[:-4]
    # Ordinary recording names pass through unchanged, so a frequency in the name still counts.
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:_MAX_NAME].rstrip("._-")
    return f"{stem or 'upload'}.wav"


def piece_list_hash(hashes: list[str]) -> str:
    """One value covering every piece in order, which both ends can work out."""
    return hashlib.sha256("".join(hashes).encode()).hexdigest()


class PartWriter:
    """One piece on its way to disk. Nothing counts until commit() has checked it."""

    def __init__(self, uploads: "Uploads", state: dict, expected_sha256: str):
        self._uploads = uploads
        self._state = state
        self._expected_sha256 = expected_sha256
        self._hash = hashlib.sha256()
        self._first_bytes = b""
        self.offset = state["received"]
        self.length = min(state["part_bytes"], state["size"] - self.offset)
        self.written = 0
        self._file = open(uploads.part_path(state["id"]), "r+b")
        self._file.seek(self.offset)
        self._file.truncate(self.offset)

    def write(self, data: bytes) -> None:
        if self.written + len(data) > self.length:
            raise UploadError(413, f"That piece is larger than the {self.length} bytes expected.", retry=True)
        if self.offset == 0 and len(self._first_bytes) < 12:
            self._first_bytes += data[: 12 - len(self._first_bytes)]
        self._hash.update(data)
        self._file.write(data)
        self.written += len(data)

    def commit(self) -> dict:
        if self.written != self.length:
            raise UploadError(422, f"That piece stopped short: {self.written} of {self.length} bytes.", retry=True)
        if self._hash.hexdigest() != self._expected_sha256:
            raise UploadError(422, "That piece arrived damaged.", retry=True)
        if self.offset == 0 and (self._first_bytes[:4] not in _WAV_MAGIC or self._first_bytes[8:12] != b"WAVE"):
            raise UploadError(415, "This file isn't really a WAV, whatever its name says.")
        self._file.flush()
        os.fsync(self._file.fileno())
        state = dict(self._state)
        state["received"] = self.offset + self.length
        state["part_hashes"] = [*state["part_hashes"], self._expected_sha256]
        state["updated_at"] = time.time()
        self._uploads.save_state(state)
        return state

    def abandon(self) -> None:
        """Cut off whatever came in, so the record and the file agree again."""
        try:
            self._file.seek(self.offset)
            self._file.truncate(self.offset)
        except OSError:
            pass  # the next piece cuts it off anyway

    def close(self) -> None:
        try:
            self._file.close()
        except OSError:
            pass
        self._uploads.release(self._state["id"])


class Uploads:
    def __init__(self, uploads_dir: str | os.PathLike, state_dir: str | os.PathLike, part_bytes: int = PART_BYTES):
        self.dir = Path(uploads_dir)
        self.state_dir = Path(state_dir)
        self.part_bytes = part_bytes
        self._lock = threading.Lock()
        self._busy: set[str] = set()

    def part_path(self, upload_id: str) -> Path:
        return self.dir / f"{upload_id}.part"

    def _state_path(self, upload_id: str) -> Path:
        return self.state_dir / f"{upload_id}.json"

    def save_state(self, state: dict) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self._state_path(state["id"])
        scratch = path.with_suffix(".tmp")
        scratch.write_text(json.dumps(state))
        os.replace(scratch, path)

    def _load(self, upload_id: str) -> dict:
        if not _ID.fullmatch(upload_id):
            raise UploadError(404, "No such upload.")
        try:
            return json.loads(self._state_path(upload_id).read_text())
        except (OSError, ValueError):
            raise UploadError(404, "No such upload.")

    @staticmethod
    def public(state: dict) -> dict:
        return {k: state[k] for k in ("id", "name", "original_name", "size", "received", "part_bytes", "part_hashes")}

    def create(self, name: str, size: int) -> dict:
        final = safe_name(name)
        if size < _MIN_WAV_BYTES:
            raise UploadError(422, "That's too small to be a WAV file.")
        self.dir.mkdir(exist_ok=True)
        free = shutil.disk_usage(self.dir).free
        if size + _FREE_MARGIN_BYTES > free:
            raise UploadError(507, f"Not enough room on the archive: {size / 2**30:.1f} GiB needed, {free / 2**30:.1f} GiB free.")
        upload_id = uuid.uuid4().hex
        self.part_path(upload_id).touch()
        now = time.time()
        state = {
            "id": upload_id,
            "name": final,
            "original_name": name[:500],
            "size": size,
            "received": 0,
            "part_bytes": self.part_bytes,
            "part_hashes": [],
            "created_at": now,
            "updated_at": now,
        }
        self.save_state(state)
        return state

    def get(self, upload_id: str) -> dict:
        return self._load(upload_id)

    def unfinished(self) -> list[dict]:
        if not self.state_dir.is_dir():
            return []
        found = []
        for path in self.state_dir.glob("*.json"):
            try:
                found.append(self._load(path.stem))
            except UploadError:
                continue
        return sorted(found, key=lambda s: s["created_at"])

    def begin_part(self, upload_id: str, offset: int, expected_sha256: str) -> PartWriter:
        state = self._load(upload_id)
        if not _SHA256.fullmatch(expected_sha256):
            raise UploadError(422, "That isn't a SHA-256 value.")
        if offset != state["received"]:
            raise UploadError(409, "That piece isn't the next one needed.", retry=True, received=state["received"])
        if offset >= state["size"]:
            raise UploadError(409, "Every piece of this file is already here.", received=state["received"])
        with self._lock:
            if upload_id in self._busy:
                # A retry can get here before the server notices the old connection died.
                raise UploadError(409, "An earlier try at this piece is still finishing.", retry=True, received=state["received"])
            self._busy.add(upload_id)
        try:
            return PartWriter(self, state, expected_sha256)
        except BaseException:
            self.release(upload_id)
            raise

    def release(self, upload_id: str) -> None:
        with self._lock:
            self._busy.discard(upload_id)

    def discard(self, upload_id: str) -> None:
        state = self._load(upload_id)
        self.part_path(state["id"]).unlink(missing_ok=True)
        self._state_path(state["id"]).unlink(missing_ok=True)

    def finish(self, upload_id: str, pieces_sha256: str) -> Path:
        """Check the whole file, then put it where the library will find it."""
        state = self._load(upload_id)
        if state["received"] != state["size"]:
            raise UploadError(409, f"Only {state['received']} of {state['size']} bytes are here.", retry=True, received=state["received"])
        if piece_list_hash(state["part_hashes"]) != pieces_sha256:
            raise UploadError(422, "The pieces here don't add up to the file that was sent.")
        part = self.part_path(upload_id)
        if part.stat().st_size != state["size"]:
            raise UploadError(422, "The file on the archive isn't the size it should be.")
        try:
            read_header(part)
        except WavError as err:
            self.discard(upload_id)
            raise UploadError(415, f"The file arrived whole, but it can't be used as a WAV: {err}")
        with self._lock:
            final = self._free_name(state["name"])
            os.rename(part, final)
        self._state_path(upload_id).unlink(missing_ok=True)
        return final

    def _free_name(self, name: str) -> Path:
        """Never overwrite: a second file with the same name gets a number instead."""
        stem = name[:-4]
        candidate = self.dir / name
        number = 2
        while candidate.exists():
            candidate = self.dir / f"{stem}-{number}.wav"
            number += 1
        return candidate

    def purge_stale(self, now: float | None = None) -> None:
        """Abandoned uploads shouldn't sit on the archive forever."""
        now = time.time() if now is None else now
        for state in self.unfinished():
            if now - state["updated_at"] > STALE_AFTER_S:
                try:
                    self.discard(state["id"])
                except OSError:
                    pass  # the share is down: this can wait until the next time
