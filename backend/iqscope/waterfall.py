"""Waterfall tiles, which cost about the same at any zoom.

A tile is 256 rows of 2048 one-byte levels, on a fixed grid:

  level  time zoom: each row covers 20 ms * 2^level
  z      frequency zoom: the view of the band is 2^z times narrower and sharper
  x      which slice of the band, from 0 to 2^z - 1
  t      which tile along the recording

Each row reads one short burst of pairs from the middle of its time span, rather than
every pair. A tile therefore reads about the same amount whether it covers 5 seconds
or 3 hours, which is what makes a 112 GiB file practical. At fine zoom the bursts
cover nearly everything; at coarse zoom they're samples, so a brief event can fall
between them and only appear after zooming in.
"""

from __future__ import annotations

import hashlib
import os
from math import ceil
from pathlib import Path

import numpy as np
from scipy.fft import fft, rfft

from .recording import Recording

TILE_ROWS = 256
BINS = 2048
BASE_ROW_S = 0.02
MAX_ZOOM = 6
DB_MIN = -130.0
DB_MAX = 0.0
_MAX_FRAMES_PER_ROW = 16

# Ingest builds whole levels only while they stay this small; zooming in further on a
# long recording computes tiles when they're first needed.
PLANNED_TILES_PER_LEVEL = 64


def row_seconds(level: int) -> float:
    return BASE_ROW_S * 2**level


def tile_seconds(level: int) -> float:
    return TILE_ROWS * row_seconds(level)


def max_level(duration_s: float) -> int:
    """The coarsest level that matters: the first where one tile covers everything."""
    level = 0
    while tile_seconds(level) < duration_s:
        level += 1
    return level


def tile_count(duration_s: float, level: int) -> int:
    return max(ceil(duration_s / tile_seconds(level)), 1)


def planned_tiles(duration_s: float) -> list[tuple[int, int]]:
    """The (level, t) pairs ingest builds, coarsest first so an overview appears early."""
    plan = []
    for level in range(max_level(duration_s), -1, -1):
        count = tile_count(duration_s, level)
        if count > PLANNED_TILES_PER_LEVEL:
            break
        plan.extend((level, t) for t in range(count))
    return plan


def compute_tile(rec: Recording, level: int, z: int, x: int, t: int) -> bytes:
    """Compute one tile. IQ shows the whole band; audio shows 0 Hz up to half its rate."""
    is_iq = rec.kind == "iq"
    fft_size = (BINS << z) if is_iq else (BINS * 2 << z)
    source_rate = rec.header.sample_rate
    row_pairs = row_seconds(level) * source_rate
    frames = int(np.clip(row_pairs // fft_size, 1, max(_MAX_FRAMES_PER_ROW >> z, 1)))
    window = np.hanning(fft_size).astype(np.float32)
    scale = 1.0 / window.sum() ** 2
    if not is_iq:
        scale *= 4.0  # a real tone splits between positive and negative frequencies

    first_bin = x * BINS
    out = np.zeros((TILE_ROWS, BINS), dtype=np.uint8)
    for row in range(TILE_ROWS):
        row_start = (t * TILE_ROWS + row) * row_pairs
        if row_start >= rec.header.frames:
            break
        burst = frames * fft_size
        start = max(int(row_start + row_pairs / 2 - burst / 2), 0)
        raw = rec.reader.read(start, burst)
        usable = raw.shape[0] // fft_size
        if usable == 0:
            continue
        samples = raw[: usable * fft_size]
        if is_iq:
            data = samples[:, 0] + 1j * samples[:, 1]
            spectrum = np.fft.fftshift(fft(data.reshape(usable, fft_size) * window, axis=1), axes=1)
        else:
            mono = samples.mean(axis=1)
            spectrum = rfft(mono.reshape(usable, fft_size) * window, axis=1)
        selected = spectrum[:, first_bin : first_bin + BINS]
        power = (selected.real**2 + selected.imag**2).mean(axis=0) * scale
        db = 10 * np.log10(np.maximum(power, 1e-30))
        out[row] = np.clip((db - DB_MIN) * 255 / (DB_MAX - DB_MIN), 0, 255).astype(np.uint8)
    return out.tobytes()


class TileStore:
    """Tiles on disk. They can always be rebuilt, so a failed write is never fatal."""

    def __init__(self, cache_dir: str | os.PathLike, max_bytes: int = 20 * 2**30):
        self.root = Path(cache_dir) / "waterfall"
        self.max_bytes = max_bytes

    @staticmethod
    def key(relpath: str, size: int, mtime_ns: int) -> str:
        """A changed recording gets a new key, which leaves its old tiles behind to expire."""
        return hashlib.sha256(f"{relpath}\0{size}\0{mtime_ns}".encode()).hexdigest()[:24]

    def _path(self, key: str, level: int, z: int, x: int, t: int) -> Path:
        return self.root / key / f"L{level}_Z{z}_X{x}_T{t}.u8"

    def has(self, key: str, level: int, z: int, x: int, t: int) -> bool:
        return self._path(key, level, z, x, t).exists()

    def get(self, key: str, level: int, z: int, x: int, t: int) -> bytes | None:
        path = self._path(key, level, z, x, t)
        try:
            data = path.read_bytes()
            os.utime(path)  # recently used tiles are the last to be removed
            return data
        except OSError:
            return None

    def put(self, key: str, level: int, z: int, x: int, t: int, data: bytes) -> None:
        path = self._path(key, level, z, x, t)
        temp = path.with_suffix(f".{os.getpid()}.{id(data)}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp.write_bytes(data)
            temp.replace(path)
        except OSError:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

    def trim(self) -> int:
        """Remove the least recently used tiles once over budget. Returns how many."""
        try:
            files = [(f.stat(), f) for f in self.root.glob("*/*.u8")]
        except OSError:
            return 0
        total = sum(stat.st_size for stat, _ in files)
        removed = 0
        for stat, path in sorted(files, key=lambda item: item[0].st_mtime):
            if total <= self.max_bytes:
                break
            try:
                path.unlink()
                total -= stat.st_size
                removed += 1
            except OSError:
                pass
        return removed


# --- The close-up spectrum: a fixed slice of the band over one second ------------------

SLICE_ROWS_PER_SECOND = 20
SLICE_WINDOW_S = 0.055
SLICE_MAX_BINS = 1024
_SLICE_FLOOR_DB = -140.0


def compute_slice(rec: Recording, second: int, center_hz: float, span_hz: float) -> tuple[dict, np.ndarray]:
    """Spectrum of one slice of the band, in rows through one second of the recording.

    This uses a long FFT of the raw pairs, with no tuning or filtering, so the result
    doesn't change as the tuner moves within the slice. center_hz is relative to the
    recording's center. Returns a description and an array of (rows, bins) in dBFS.
    """
    rate = rec.header.sample_rate
    fft_size = 2 ** max(round(np.log2(SLICE_WINDOW_S * rate)), 8)
    bin_hz = rate / fft_size
    first_bin = round((center_hz - span_hz / 2 + rate / 2) / bin_hz)
    bin_count = max(round(span_hz / bin_hz), 1)
    group = max(ceil(bin_count / SLICE_MAX_BINS), 1)  # wide slices keep each group's peak
    bins_out = ceil(bin_count / group)

    window = np.hanning(fft_size).astype(np.float32)
    scale = 1.0 / window.sum() ** 2
    hop = rate / SLICE_ROWS_PER_SECOND
    start_pair = second * rate
    available = min(rate, rec.header.frames - start_pair)
    rows = max(ceil(available / hop), 1)
    out = np.full((rows, bins_out), _SLICE_FLOOR_DB, dtype=np.float32)

    for row in range(rows):
        middle = start_pair + (row + 0.5) * hop
        begin = int(middle - fft_size / 2)
        raw = rec.reader.read(max(begin, 0), fft_size - max(-begin, 0))
        if raw.shape[0] == 0:
            continue
        z = np.zeros(fft_size, dtype=np.complex64)
        z[max(-begin, 0) : max(-begin, 0) + raw.shape[0]] = raw[:, 0] + 1j * raw[:, 1]
        spectrum = np.fft.fftshift(fft(z * window))
        power = (spectrum.real**2 + spectrum.imag**2) * scale

        low = max(first_bin, 0)
        high = min(first_bin + bin_count, fft_size)
        if high <= low:
            continue
        chosen = power[low:high]
        # Bins beyond the recording's edges simply stay at the floor.
        offset = low - first_bin
        group_index = (np.arange(chosen.size) + offset) // group
        peaks = np.zeros(bins_out)
        np.maximum.at(peaks, group_index, chosen)
        out[row] = 10 * np.log10(np.maximum(peaks, 1e-14))

    return (
        {
            "second": second,
            "rows": rows,
            "bins": bins_out,
            "rows_per_second": SLICE_ROWS_PER_SECOND,
            "first_hz": -rate / 2 + first_bin * bin_hz + (group - 1) * bin_hz / 2,
            "bin_hz": group * bin_hz,
            "center_hz": center_hz,
            "span_hz": span_hz,
        },
        out,
    )
