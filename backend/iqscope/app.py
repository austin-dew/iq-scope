"""The web app: API plus the built frontend.

The health check needs care, because of how the share reaches the container. The
container's /radio is always a mount point, even when the archive share is down: it's
then a bind of the host's empty directory. So "is it mounted?" can't be answered by
asking whether /radio is a mount point. Instead, look at the filesystem type behind it.

A dead network share can also block a read for many seconds, so reads run on separate
threads with time limits, and the app stays responsive whatever the share does.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import struct
import sys
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.requests import ClientDisconnect

from .dsp import AUDIO_RATE, GAIN_BLOCK, MODES, SSB_LOW_HZ, TuningError, audio_chunk, demodulate_chunk
from .library import Entry, Library
from .recording import NotIQError
from .uploads import PART_BYTES, Uploads, UploadError
from .wavfile import WavError
from .waterfall import BINS, MAX_ZOOM, TILE_ROWS, TileStore, compute_slice, max_level, tile_count

_STATIC = Path(__file__).parent / "static"
_NETWORK_FILESYSTEMS = {"cifs", "smb3"}
_PROBE_TIMEOUT_S = 3.0
_CHUNK_CACHE_BYTES = 256 * 2**20
_RAW_MAX_PAIRS = 4096
_UPLOAD_WRITE_BYTES = 2**20

_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; font-src 'self'; worker-src 'self'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'"
)


def filesystem_type(path: str, mountinfo: str = "/proc/self/mountinfo") -> str | None:
    """The filesystem at exactly this mount point, or None if it isn't one.

    Each line reads: id parent dev root MOUNTPOINT options [tags] - FSTYPE source opts.
    A later line for the same point is mounted over an earlier one, so the last one wins.
    """
    found = None
    with open(mountinfo, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            fields = line.split()
            if len(fields) > 4 and fields[4] == path and " - " in line:
                found = line.split(" - ", 1)[1].split()[0]
    return found


class ShareProbe:
    """Checks the share without ever letting a hung read pile up threads."""

    def __init__(self, radio_dir: str, mountinfo: str = "/proc/self/mountinfo"):
        self.radio_dir = radio_dir
        self.mountinfo = mountinfo
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="share-probe")
        self._pending: Future | None = None

    def _read_recordings(self) -> bool:
        return os.path.isdir(os.path.join(self.radio_dir, "recordings")) and isinstance(
            os.listdir(os.path.join(self.radio_dir, "recordings")), list
        )

    def check(self) -> dict:
        fstype = filesystem_type(self.radio_dir, self.mountinfo)
        if fstype not in _NETWORK_FILESYSTEMS:
            return {
                "mounted": False,
                "readable": False,
                "detail": f"{self.radio_dir} is not on the archive share (filesystem: {fstype or 'none'}).",
            }

        # A read that's still stuck from earlier means the share is still not answering.
        if self._pending is None or self._pending.done():
            self._pending = self._pool.submit(self._read_recordings)
        try:
            readable = self._pending.result(timeout=_PROBE_TIMEOUT_S)
        except FutureTimeout:
            return {
                "mounted": True,
                "readable": False,
                "detail": f"The share didn't answer within {_PROBE_TIMEOUT_S:.0f} s.",
            }
        except OSError as err:
            return {"mounted": True, "readable": False, "detail": f"Reading the share failed: {err.strerror}."}

        if not readable:
            return {"mounted": True, "readable": False, "detail": "The share has no recordings folder."}
        return {"mounted": True, "readable": True, "detail": "The archive share is online."}


class ResultCache:
    """Runs expensive work on a small pool, sharing it between identical requests, and
    keeps recent results up to a byte budget. Identical settings always give identical
    results, so nothing here can go stale."""

    def __init__(self, workers: int, max_bytes: int):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="iqscope-work")
        self._max_bytes = max_bytes
        self._results: OrderedDict[tuple, bytes] = OrderedDict()
        self._bytes = 0
        self._running: dict[tuple, Future] = {}
        self._lock = threading.Lock()

    def submit(self, key: tuple, work) -> Future:
        with self._lock:
            if key in self._results:
                self._results.move_to_end(key)
                done: Future = Future()
                done.set_result(self._results[key])
                return done
            if key in self._running:
                return self._running[key]
            future = self._pool.submit(work)
            self._running[key] = future
        future.add_done_callback(lambda f: self._finish(key, f))
        return future

    def _finish(self, key: tuple, future: Future) -> None:
        with self._lock:
            self._running.pop(key, None)
            if future.cancelled() or future.exception() is not None:
                return
            data = future.result()
            self._results[key] = data
            self._bytes += len(data)
            while self._bytes > self._max_bytes and self._results:
                _, old = self._results.popitem(last=False)
                self._bytes -= len(old)


def encode_frame(header: dict, arrays: dict[str, np.ndarray]) -> bytes:
    """A binary frame: header length (4 bytes), JSON header, then Float32 arrays.

    The JSON is padded to a multiple of 4 bytes, so a browser can view the arrays
    directly as Float32Array without copying them.
    """
    header = {**header, "arrays": [{"name": n, "length": int(a.size)} for n, a in arrays.items()]}
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    parts = [struct.pack("<I", len(text)), text]
    parts.extend(a.astype("<f4").tobytes() for a in arrays.values())
    return b"".join(parts)


def _audio_frame(entry: Entry, index: int) -> bytes:
    """An audio recording's chunk: only sound, as there are no pairs behind it."""
    with entry.open() as rec:
        audio = audio_chunk(rec, index)
        source_rate = rec.header.sample_rate
    header = {
        "index": index,
        "mode": "audio",
        "source_rate": source_rate,
        "iq_rate": AUDIO_RATE,
        "audio_rate": AUDIO_RATE,
        "pairs_per_sample": 1,
        "start_pair": index * source_rate,
        "gain_block": GAIN_BLOCK,
        "offset_hz": 0,
        "bw_hz": 0,
    }
    return encode_frame(header, {"audio": audio})


def _chunk_frame(entry: Entry, index: int, offset_hz: int, mode: str, bw_hz: float | None, fields: tuple[str, ...]) -> bytes:
    with entry.open() as rec:
        r = demodulate_chunk(rec, index, offset_hz, mode, bw_hz)
    available = {
        "audio": r.audio,
        "i": r.i,
        "q": r.q,
        "demod": r.demod,
        "gain_nodes": r.gain_nodes,
        "carrier": r.carrier,
        "discriminator": r.discriminator,
    }
    arrays = {name: available[name] for name in fields if available.get(name) is not None}
    header = {
        "index": r.index,
        "mode": r.mode,
        "source_rate": r.source_rate,
        "iq_rate": r.iq_rate,
        "audio_rate": r.audio_rate,
        "pairs_per_sample": r.pairs_per_sample,
        "start_pair": r.start_pair,
        "gain_block": GAIN_BLOCK,
        "offset_hz": offset_hz,
        "bw_hz": MODES[mode].bandwidth_hz if bw_hz is None else bw_hz,
    }
    return encode_frame(header, arrays)


def _slice_frame(entry: Entry, second: int, center_hz: int, span_hz: int) -> bytes:
    with entry.open() as rec:
        header, rows = compute_slice(rec, second, center_hz, span_hz)
    return encode_frame(header, {"db": rows})


def _raw_frame(entry: Entry, start: int, count: int) -> bytes:
    """Pairs exactly as recorded, before any tuning, for Analysis Mode's constellation."""
    with entry.open() as rec:
        pairs = rec.reader.read(start, count)
        rate = rec.header.sample_rate
    header = {"start_pair": start, "count": int(pairs.shape[0]), "sample_rate": rate}
    return encode_frame(header, {"i": pairs[:, 0], "q": pairs[:, 1]})


class Marker(BaseModel):
    id: str = Field(default="", max_length=64)
    t: float = Field(ge=0, lt=1e9)
    label: str = Field(default="", max_length=200)
    kind: Literal["note", "frequency"] = "note"
    frequency_hz: int | None = Field(default=None, gt=0, lt=300_000_000_000)

    @model_validator(mode="after")
    def frequency_marks_need_a_frequency(self) -> "Marker":
        if self.kind == "frequency" and self.frequency_hz is None:
            raise ValueError("a frequency mark needs a frequency")
        return self


class UploadStart(BaseModel):
    name: str = Field(min_length=1, max_length=500)
    size: int = Field(gt=0, lt=2**50)


class UploadFinish(BaseModel):
    pieces_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def same_page_only(x_iqscope: str | None = Header(default=None)) -> None:
    """Another website can make a browser send a plain POST here, but not with this header."""
    if x_iqscope != "1":
        raise HTTPException(403, "Requests that change things must come from this app's own page.")


class SidecarUpdate(BaseModel):
    """Only the fields actually sent change. Sending null clears one."""

    kind: Literal["iq", "audio"] | None = None
    frequency_hz: int | None = Field(default=None, gt=0, lt=300_000_000_000)
    markers: list[Marker] | None = Field(default=None, max_length=2000)


def create_app(
    radio_dir: str | None = None,
    cache_dir: str | None = None,
    mountinfo: str = "/proc/self/mountinfo",
    static_dir: str | os.PathLike | None = _STATIC,
    poll_s: float = 30.0,
    background: bool = True,
    upload_part_bytes: int = PART_BYTES,
) -> FastAPI:
    radio_dir = radio_dir or os.environ.get("IQSCOPE_RADIO_DIR", "/radio")
    cache_dir = cache_dir or os.environ.get("IQSCOPE_CACHE_DIR", "/cache")
    probe = ShareProbe(radio_dir, mountinfo)
    library = Library(
        os.path.join(radio_dir, "recordings"),
        TileStore(cache_dir),
        poll_s=poll_s,
        examples_dir=os.path.join(cache_dir, "examples"),
    )
    uploads = Uploads(os.path.join(radio_dir, "recordings", "uploads"), os.path.join(cache_dir, "uploads"), upload_part_bytes)
    # Slow writes to the share wait here, never in the way of anything else the app does.
    upload_writes = ThreadPoolExecutor(max_workers=2, thread_name_prefix="iqscope-upload")
    # Demodulation and tiles get separate pools, so a slow tile can't delay sound.
    chunks = ResultCache(workers=2, max_bytes=_CHUNK_CACHE_BYTES)
    tiles = ResultCache(workers=1, max_bytes=64 * 2**20)
    slices = ResultCache(workers=1, max_bytes=64 * 2**20)  # a picture must never delay sound
    raw = ResultCache(workers=1, max_bytes=8 * 2**20)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if background:
            library.start()
        yield
        if background:
            library.stop()

    app = FastAPI(title="CONELRAD SIGNAL ANALYZER", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.library = library

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = _CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def find(entry_id: str) -> Entry:
        entry = library.get(entry_id)
        if entry is None:
            raise HTTPException(404, "No such recording.")
        return entry

    @app.get("/api/health")
    def health() -> dict:
        # The app itself is fine even when the share isn't, so this always returns 200.
        # Restarting the container wouldn't bring the NAS back.
        share = probe.check()
        return {
            "status": "ok" if share["readable"] else "degraded",
            "share": share,
            "cache_writable": os.access(cache_dir, os.W_OK),
            "python": sys.version.split()[0],
        }

    @app.get("/api/modes")
    def modes() -> dict:
        return {
            "ssb_low_hz": SSB_LOW_HZ,
            "modes": {name: {"default_bw_hz": m.bandwidth_hz, "iq_rate": m.iq_rate} for name, m in MODES.items()},
        }

    @app.get("/api/recordings")
    def recordings() -> dict:
        return {
            "online": library.online,
            "complete": library.complete,
            "arriving": library.arriving,
            "recordings": [e.as_json() for e in library.list()],
        }

    @app.post("/api/rescan", dependencies=[Depends(same_page_only)])
    def rescan() -> dict:
        library.scan(manual=True)
        return {"online": library.online}

    def writable_share() -> None:
        share = probe.check()
        if not share["readable"]:
            raise UploadError(503, f"The archive is offline: {share['detail']}", retry=True)

    @app.exception_handler(UploadError)
    async def upload_errors(_request, exc: UploadError) -> JSONResponse:
        return JSONResponse({"detail": exc.message, "retry": exc.retry, **exc.extra}, status_code=exc.status)

    def share_errors(err: OSError) -> UploadError:
        if err.errno == errno.ENOSPC:
            return UploadError(507, "The archive ran out of room.")
        return UploadError(503, f"Couldn't write to the archive: {err.strerror or err}.", retry=True)

    @app.get("/api/uploads")
    def list_uploads() -> dict:
        uploads.purge_stale()
        return {"uploads": [Uploads.public(s) for s in uploads.unfinished()]}

    @app.post("/api/uploads", dependencies=[Depends(same_page_only)], status_code=201)
    def start_upload(start: UploadStart) -> dict:
        writable_share()
        try:
            return Uploads.public(uploads.create(start.name, start.size))
        except OSError as err:
            raise share_errors(err)

    @app.get("/api/uploads/{upload_id}")
    def get_upload(upload_id: str) -> dict:
        return Uploads.public(uploads.get(upload_id))

    @app.delete("/api/uploads/{upload_id}", dependencies=[Depends(same_page_only)], status_code=204)
    def cancel_upload(upload_id: str) -> None:
        try:
            uploads.discard(upload_id)
        except OSError as err:
            raise share_errors(err)

    @app.put("/api/uploads/{upload_id}/part", dependencies=[Depends(same_page_only)])
    async def upload_part(upload_id: str, request: Request, offset: int = Query(ge=0), sha256: str = Query(max_length=64)) -> dict:
        loop = asyncio.get_running_loop()
        run = lambda *args: loop.run_in_executor(upload_writes, *args)  # noqa: E731
        try:
            writer = await run(uploads.begin_part, upload_id, offset, sha256)
        except OSError as err:
            raise share_errors(err)
        committed = False
        try:
            declared = request.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > writer.length:
                raise UploadError(413, f"That piece is larger than the {writer.length} bytes expected.", retry=True)
            # The body goes straight to the file: at most a little of it is ever in memory.
            pending = bytearray()
            async for data in request.stream():
                pending += data
                if len(pending) >= _UPLOAD_WRITE_BYTES:
                    await run(writer.write, bytes(pending))
                    pending.clear()
            if pending:
                await run(writer.write, bytes(pending))
            state = await run(writer.commit)
            committed = True
            return {"received": state["received"]}
        except ClientDisconnect:
            raise UploadError(400, "The connection dropped during that piece.", retry=True)
        except OSError as err:
            raise share_errors(err)
        finally:
            if not committed:
                await run(writer.abandon)
            await run(writer.close)

    @app.post("/api/uploads/{upload_id}/finish", dependencies=[Depends(same_page_only)])
    def finish_upload(upload_id: str, finish: UploadFinish) -> dict:
        try:
            path = uploads.finish(upload_id, finish.pieces_sha256)
            return library.add_finished(path).as_json()
        except OSError as err:
            raise share_errors(err)

    @app.get("/api/recordings/{entry_id}")
    def recording(entry_id: str) -> dict:
        return find(entry_id).as_json()

    @app.get("/api/recordings/{entry_id}/sidecar")
    def get_sidecar(entry_id: str) -> dict:
        entry = find(entry_id)
        return {
            "kind": entry.kind_override,
            "frequency_hz": entry.info.get("user_frequency_hz"),
            "markers": entry.info.get("markers", []),
        }

    @app.put("/api/recordings/{entry_id}/sidecar")
    def put_sidecar(entry_id: str, update: SidecarUpdate) -> dict:
        entry = find(entry_id)
        if update.kind == "iq" and entry.info.get("channels") != 2:
            raise HTTPException(422, "IQ needs exactly two channels.")
        try:
            library.update_sidecar(entry, update.model_dump(exclude_unset=True, exclude_none=False))
        except OSError as err:
            raise HTTPException(503, f"Couldn't save to the share: {err.strerror}.")
        return entry.as_json()

    @app.get("/api/recordings/{entry_id}/waterfall")
    async def waterfall(
        entry_id: str,
        level: int = Query(ge=0, le=40),
        t: int = Query(ge=0),
        z: int = Query(default=0, ge=0, le=MAX_ZOOM),
        x: int = Query(default=0, ge=0),
    ) -> Response:
        entry = find(entry_id)
        if entry.kind not in ("iq", "audio"):
            raise HTTPException(409, f"This recording is {entry.kind}; choose what it is first.")
        duration = entry.info["duration_s"]
        if level > max_level(duration) or t >= tile_count(duration, level) or x >= 2**z:
            raise HTTPException(404, "That tile is outside this recording.")
        key = ("tile", entry.cache_key, level, z, x, t)
        try:
            data = await asyncio.wrap_future(tiles.submit(key, lambda: library.tile(entry, level, z, x, t)))
        except (OSError, WavError) as err:
            raise HTTPException(503, f"Couldn't read the recording: {err}.")
        return Response(
            data,
            media_type="application/octet-stream",
            headers={"X-Tile-Rows": str(TILE_ROWS), "X-Tile-Bins": str(BINS), "Cache-Control": "private, max-age=3600"},
        )

    @app.get("/api/recordings/{entry_id}/slice")
    async def band_slice(
        entry_id: str,
        index: int = Query(ge=0),
        center_hz: int = Query(ge=-1_000_000_000, le=1_000_000_000),
        span_hz: int = Query(ge=500, le=10_000_000),
    ) -> Response:
        entry = find(entry_id)
        if entry.kind != "iq":
            raise HTTPException(409, "AUDIO RECORDING: NO IQ DATA. There's no band to look at.")
        rate = entry.info["sample_rate"]
        if index >= int(np.ceil(entry.info["duration_s"])):
            raise HTTPException(404, "That second is outside this recording.")
        if span_hz > rate or abs(center_hz) > rate / 2:
            raise HTTPException(422, f"That slice doesn't fit in a {rate:,} S/s recording.")
        key = ("slice", entry.cache_key, index, center_hz, span_hz)
        try:
            data = await asyncio.wrap_future(slices.submit(key, lambda: _slice_frame(entry, index, center_hz, span_hz)))
        except (OSError, WavError) as err:
            raise HTTPException(503, f"Couldn't read the recording: {err}.")
        return Response(data, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})

    @app.get("/api/recordings/{entry_id}/raw")
    async def raw_pairs(
        entry_id: str,
        start: int = Query(ge=0),
        count: int = Query(ge=1, le=_RAW_MAX_PAIRS),
    ) -> Response:
        entry = find(entry_id)
        if entry.kind != "iq":
            raise HTTPException(409, "AUDIO RECORDING: NO IQ DATA. There are no pairs to show.")
        if start >= entry.info["frames"]:
            raise HTTPException(404, "That pair is outside this recording.")
        key = ("raw", entry.cache_key, start, count)
        try:
            data = await asyncio.wrap_future(raw.submit(key, lambda: _raw_frame(entry, start, count)))
        except (OSError, WavError) as err:
            raise HTTPException(503, f"Couldn't read the recording: {err}.")
        return Response(data, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})

    @app.get("/api/recordings/{entry_id}/chunk")
    async def chunk(
        entry_id: str,
        index: int = Query(ge=0),
        offset_hz: int = Query(ge=-1_000_000_000, le=1_000_000_000),
        mode: Literal["am", "usb", "lsb", "cw", "nfm", "wfm"] = "am",
        bw_hz: float | None = Query(default=None, gt=0, le=1_000_000),
        fields: str = Query(default="audio,gain_nodes", max_length=80),
    ) -> Response:
        entry = find(entry_id)
        if entry.kind not in ("iq", "audio"):
            raise HTTPException(409, f"This recording is {entry.kind}; choose what it is first.")
        names = tuple(sorted(set(fields.split(","))))
        allowed = {"audio", "i", "q", "demod", "gain_nodes", "carrier", "discriminator"}
        if not set(names) <= allowed:
            raise HTTPException(422, "Unknown field requested.")
        if entry.kind == "audio":
            key = ("audio", entry.cache_key, index)
            work = lambda: _audio_frame(entry, index)  # noqa: E731
        else:
            key = ("chunk", entry.cache_key, index, offset_hz, mode, bw_hz, names)
            work = lambda: _chunk_frame(entry, index, offset_hz, mode, bw_hz, names)  # noqa: E731
        try:
            data = await asyncio.wrap_future(chunks.submit(key, work))
        except TuningError as err:
            raise HTTPException(422, str(err))
        except IndexError as err:
            raise HTTPException(404, str(err))
        except NotIQError as err:
            raise HTTPException(409, str(err))
        except (OSError, WavError) as err:
            raise HTTPException(503, f"Couldn't read the recording: {err}.")
        return Response(data, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})

    @app.exception_handler(HTTPException)
    async def plain_errors(_request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    if static_dir is not None and Path(static_dir, "index.html").exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


app = create_app()
