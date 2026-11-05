"""The list of recordings: watching the folder, sidecars and background ingest.

Clients only ever see opaque ids. A filesystem path never comes in from outside, so a
request can't name anything except a file the scan itself found.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .examples import ensure_examples, example_info
from .recording import Recording, open_recording
from .wavfile import WavError
from .waterfall import TileStore, compute_tile, max_level, planned_tiles
from . import waterfall as wf

log = logging.getLogger("iqscope.library")

SIDECAR_SUFFIX = ".iqscope.json"
_SIDECAR_MAX_BYTES = 1 << 20
# A file this long untouched isn't still being copied, so a restart needn't wait for it.
_SETTLED_AGE_S = 120.0
_MAX_FREQUENCY_HZ = 300_000_000_000
_MAX_MARKERS = 2000
_MAX_LABEL = 200
_EXAMPLES_PREFIX = "@examples/"


def clean_markers(raw: object) -> list[dict]:
    """Keep only well-formed markers, in time order. Sidecars are untrusted."""
    if not isinstance(raw, list):
        return []
    markers = []
    for item in raw[:_MAX_MARKERS]:
        if not isinstance(item, dict):
            continue
        t = item.get("t")
        kind = item.get("kind", "note")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t < 1e9 or kind not in ("note", "frequency"):
            continue
        marker = {
            "id": str(item.get("id", ""))[:64],
            "t": float(t),
            "label": str(item.get("label", ""))[:_MAX_LABEL],
            "kind": kind,
        }
        if kind == "frequency":
            frequency = item.get("frequency_hz")
            if isinstance(frequency, bool) or not isinstance(frequency, int) or not 0 < frequency < _MAX_FREQUENCY_HZ:
                continue
            marker["frequency_hz"] = frequency
        markers.append(marker)
    return sorted(markers, key=lambda m: m["t"])


def read_sidecar(path: Path) -> dict:
    """Read a sidecar, keeping only fields that are valid. Its contents are untrusted."""
    sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
    try:
        if sidecar.stat().st_size > _SIDECAR_MAX_BYTES:
            return {}
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    clean = {}
    if data.get("kind") in ("iq", "audio"):
        clean["kind"] = data["kind"]
    frequency = data.get("frequency_hz")
    if isinstance(frequency, int) and not isinstance(frequency, bool) and 0 < frequency < _MAX_FREQUENCY_HZ:
        clean["frequency_hz"] = frequency
    if "markers" in data:
        clean["markers"] = clean_markers(data["markers"])
    return clean


def write_sidecar(path: Path, values: dict) -> None:
    """Merge values into a sidecar, without disturbing fields this version doesn't know."""
    sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
    try:
        existing = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            existing = {}
    except (OSError, ValueError):
        existing = {}
    for name, value in values.items():
        if value is None:
            existing.pop(name, None)
        else:
            existing[name] = value
    temp = sidecar.with_name(sidecar.name + ".part")
    temp.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    temp.replace(sidecar)


@dataclass
class Entry:
    id: str
    path: Path
    relpath: str
    size: int
    mtime_ns: int
    info: dict = field(default_factory=dict)
    kind_override: str | None = None
    available: bool = True
    tiles_done: int = 0
    tiles_total: int = 0

    @property
    def is_example(self) -> bool:
        return self.relpath.startswith(_EXAMPLES_PREFIX)

    @property
    def kind(self) -> str:
        return self.info.get("kind", "unreadable")

    @property
    def cache_key(self) -> str:
        return TileStore.key(self.relpath, self.size, self.mtime_ns)

    def open(self) -> Recording:
        return open_recording(self.path, kind_override=self.kind_override)

    def as_json(self) -> dict:
        return {
            "id": self.id,
            "name": self.path.name,
            "folder": "" if "/" not in self.relpath or self.is_example else str(Path(self.relpath).parent),
            "example": example_info(self.path.name) if self.is_example else None,
            "size_bytes": self.size,
            "available": self.available,
            "ingest": {"done": self.tiles_done, "total": self.tiles_total},
            **self.info,
        }


class Library:
    def __init__(
        self,
        recordings_dir: str | os.PathLike,
        store: TileStore,
        poll_s: float = 30.0,
        examples_dir: str | os.PathLike | None = None,
    ):
        self.recordings_dir = Path(recordings_dir)
        self.examples_dir = Path(examples_dir) if examples_dir is not None else None
        self.store = store
        self.poll_s = poll_s
        self.online = False
        self.arriving: list[dict] = []  # files seen but still changing
        self.complete = False  # the list reflects a finished look at the share
        self._scan_lock = threading.Lock()
        self._entries: dict[str, Entry] = {}
        self._seen: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # --- Queries ---------------------------------------------------------------

    def list(self) -> list[Entry]:
        with self._lock:
            return sorted(self._entries.values(), key=lambda e: (e.is_example, e.relpath.lower()))

    def get(self, entry_id: str) -> Entry | None:
        with self._lock:
            return self._entries.get(entry_id)

    # --- Scanning --------------------------------------------------------------

    def _walk(self, top: Path) -> dict[str, os.stat_result]:
        def fail(err: OSError) -> None:
            raise err

        found = {}
        for root, dirs, files in os.walk(top, onerror=fail, followlinks=False):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in files:
                if name.startswith(".") or not name.lower().endswith(".wav"):
                    continue
                path = os.path.join(root, name)
                stat = os.lstat(path)
                if not os.path.islink(path):
                    found[path] = stat
        return found

    def scan(self, now: float | None = None, manual: bool = False) -> None:
        """Bring the list up to date. A file joins once its size has stopped changing.

        A manual scan can come any moment after the last one, when a file being copied may
        just not have changed yet, so it doesn't count as a look at whether it's settled.
        """
        now = time.time() if now is None else now
        with self._scan_lock:
            self._scan(now, manual)

    def _scan(self, now: float, manual: bool) -> None:
        # Examples live on this machine, so they keep working when the share is down.
        found: dict[str, os.stat_result] = {}
        if self.examples_dir is not None and self.examples_dir.is_dir():
            try:
                found.update(self._walk(self.examples_dir))
            except OSError:
                pass
        try:
            found.update(self._walk(self.recordings_dir))
            online = True
        except OSError:
            online = False

        with self._lock:
            self.online = online
            known = {str(e.path): e for e in self._entries.values()}
            for path, entry in known.items():
                if entry.is_example or online:
                    entry.available = True
                    if path not in found:
                        del self._entries[entry.id]
                else:
                    entry.available = False

        arriving = []
        for path, stat in found.items():
            signature = (stat.st_size, stat.st_mtime_ns)
            settled = self._seen.get(path) == signature or now - stat.st_mtime > _SETTLED_AGE_S
            if not manual:
                self._seen[path] = signature
            entry = known.get(path)
            if entry is not None and (entry.size, entry.mtime_ns) == signature:
                self._refresh_sidecar(entry)
                continue
            if settled:
                self._add(Path(path), stat)
            else:
                arriving.append({"name": Path(path).name, "size_bytes": stat.st_size})
        self.arriving = sorted(arriving, key=lambda a: a["name"])
        self.complete = online
        for path in set(self._seen) - set(found):
            del self._seen[path]
        self._wake.set()

    def add_finished(self, path: Path) -> Entry:
        """A file this app has just finished writing needn't wait to be noticed."""
        stat = os.lstat(path)
        with self._scan_lock:
            self._seen[str(path)] = (stat.st_size, stat.st_mtime_ns)
            self._add(path, stat)
        self._wake.set()
        return next(e for e in self.list() if e.path == path)

    def _add(self, path: Path, stat: os.stat_result) -> None:
        if self.examples_dir is not None and path.is_relative_to(self.examples_dir):
            relpath = _EXAMPLES_PREFIX + path.relative_to(self.examples_dir).as_posix()
        else:
            relpath = path.relative_to(self.recordings_dir).as_posix()
        entry = Entry(
            id=hashlib.sha256(relpath.encode()).hexdigest()[:16],
            path=path,
            relpath=relpath,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )
        self._load_info(entry)
        with self._lock:
            self._entries[entry.id] = entry

    def _refresh_sidecar(self, entry: Entry) -> None:
        sidecar = read_sidecar(entry.path)
        changed = (
            sidecar.get("kind") != entry.kind_override
            or sidecar.get("frequency_hz") != entry.info.get("user_frequency_hz")
            or sidecar.get("markers", []) != entry.info.get("markers", [])
        )
        if changed:
            self._load_info(entry)

    def _load_info(self, entry: Entry) -> None:
        """Parse a recording's header and combine it with the user's sidecar answers."""
        sidecar = read_sidecar(entry.path)
        entry.kind_override = sidecar.get("kind")
        try:
            with entry.open() as rec:
                entry.info = self._describe(rec, sidecar)
        except (WavError, ValueError) as err:
            entry.info = {"kind": "unreadable", "kind_reason": str(err), "warnings": []}
        except OSError as err:
            entry.info = {"kind": "unreadable", "kind_reason": err.strerror or "read failed", "warnings": []}
        if entry.kind in ("iq", "audio"):
            plan = planned_tiles(entry.info["duration_s"])
            entry.tiles_total = len(plan)
            entry.tiles_done = sum(self.store.has(entry.cache_key, level, 0, 0, t) for level, t in plan)
        else:
            entry.tiles_total = entry.tiles_done = 0

    @staticmethod
    def _describe(rec: Recording, sidecar: dict) -> dict:
        h = rec.header
        frequency, source = rec.frequency_hz, rec.frequency_source
        user_frequency = sidecar.get("frequency_hz")
        if user_frequency is not None:
            frequency, source = user_frequency, "set by the user"
        rate = h.sample_rate
        is_iq = rec.kind == "iq"
        return {
            "kind": rec.kind,
            "kind_reason": rec.kind_reason,
            "container": h.container,
            "sample_type": h.sample_type,
            "channels": h.channels,
            "sample_rate": rate,
            "frames": h.frames,
            "duration_s": h.duration_s,
            "frequency_hz": frequency,
            "user_frequency_hz": user_frequency,
            "frequency_source": source,
            "frequency_note": None if user_frequency else rec.frequency_note,
            "started_at": rec.started_at.isoformat() if rec.started_at else None,
            "recorded_with": h.aux_recorded_with,
            "lr_match": rec.lr_match,
            "warnings": list(h.warnings),
            "markers": sidecar.get("markers", []),
            "waterfall": {
                "bins": wf.BINS,
                "tile_rows": wf.TILE_ROWS,
                "base_row_s": wf.BASE_ROW_S,
                "max_level": max_level(h.duration_s),
                "max_zoom": wf.MAX_ZOOM,
                "db_min": wf.DB_MIN,
                "db_max": wf.DB_MAX,
                "low_hz": -rate / 2 if is_iq else 0.0,
                "span_hz": float(rate) if is_iq else rate / 2,
            },
        }

    def update_sidecar(self, entry: Entry, values: dict) -> None:
        write_sidecar(entry.path, values)
        with self._lock:
            self._load_info(entry)
        self._wake.set()

    # --- Tiles -----------------------------------------------------------------

    def tile(self, entry: Entry, level: int, z: int, x: int, t: int) -> bytes:
        """Return a tile, computing and storing it the first time it's needed."""
        key = entry.cache_key
        data = self.store.get(key, level, z, x, t)
        if data is None:
            with entry.open() as rec:
                data = compute_tile(rec, level, z, x, t)
            self.store.put(key, level, z, x, t, data)
        return data

    def _ingest_one(self, entry: Entry) -> bool:
        """Build the next planned tile for this recording. False means nothing was left."""
        if not entry.available or entry.kind not in ("iq", "audio"):
            return False
        for level, t in planned_tiles(entry.info["duration_s"]):
            if self.store.has(entry.cache_key, level, 0, 0, t):
                continue
            try:
                self.tile(entry, level, 0, 0, t)
            except OSError:
                return False  # the share is struggling; the next scan will try again
            entry.tiles_done = min(entry.tiles_done + 1, entry.tiles_total)
            return True
        entry.tiles_done = entry.tiles_total
        return False

    def prepare_examples(self) -> None:
        """Make the example clips if they're missing, then list them straight away."""
        if self.examples_dir is None:
            return
        try:
            if ensure_examples(self.examples_dir):
                self.scan()
                self.scan()  # a file joins once two scans agree on its size
        except OSError:
            log.exception("couldn't generate the example clips")

    def ingest_pending(self) -> bool:
        """Do one small piece of waiting work. True if there may be more."""
        for entry in self.list():
            if self._ingest_one(entry):
                return True
        return False

    # --- Background threads ------------------------------------------------------

    def _scan_loop(self) -> None:
        while not self._stop.is_set():
            self.scan()
            self.store.trim()
            self._stop.wait(self.poll_s)

    def _ingest_loop(self) -> None:
        # On Linux this lowers only this thread, so playback keeps normal priority, and
        # everything else on the machine comes ahead of it. Disk priority wouldn't help here: the reads happen
        # on the NAS, not on this machine.
        try:
            os.setpriority(os.PRIO_PROCESS, 0, 15)
        except OSError:
            pass
        self.prepare_examples()
        while not self._stop.is_set():
            try:
                worked = self.ingest_pending()
            except Exception:  # a bad file must never stop ingest for the others
                log.exception("ingest failed")
                worked = False
            if not worked:
                self._wake.wait(self.poll_s)
                self._wake.clear()

    def start(self) -> None:
        for target, name in ((self._scan_loop, "scan"), (self._ingest_loop, "ingest")):
            thread = threading.Thread(target=target, name=f"iqscope-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        for thread in self._threads:
            thread.join(timeout=2)
        self._threads.clear()
