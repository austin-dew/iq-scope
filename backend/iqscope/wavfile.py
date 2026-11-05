"""WAV header parser and random-access sample reader.

A WAV is a list of chunks: a 4-byte id, a size, then that many bytes. Samples live
in the `data` chunk, which is not always at byte 44, so the parser walks the list.

Ordinary RIFF stores sizes in 32 bits, capping a file at 4 GiB. RF64 and BW64 set the
size fields to 0xFFFFFFFF and put the real 64-bit sizes in a `ds64` chunk near the start.

Recorders can also leave wrong sizes behind (a crash, or a counter that wrapped at
4 GiB), so sizes are checked against the actual file and every correction is reported
in `WavHeader.warnings`. Samples are fetched with pread, so no read touches more of
the file than was asked for, however long the recording is.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_FORMAT_PCM = 1
_FORMAT_FLOAT = 3
_FORMAT_EXTENSIBLE = 0xFFFE
_SIZE_UNKNOWN = 0xFFFFFFFF
_MAX_CHUNKS = 4096
_MAX_METADATA_BYTES = 64 * 1024

# Upper limit for one read. Generous for a 1 s chunk at any SDR rate, but stops a
# bad request from allocating all of the host's memory.
MAX_READ_FRAMES = 1 << 26

# (format tag, bits per sample) -> sample type name
_SAMPLE_TYPES = {
    (_FORMAT_PCM, 8): "u8",
    (_FORMAT_PCM, 16): "i16",
    (_FORMAT_PCM, 24): "i24",
    (_FORMAT_PCM, 32): "i32",
    (_FORMAT_FLOAT, 32): "f32",
}


class WavError(ValueError):
    """The file cannot be used as a WAV recording."""


@dataclass(frozen=True)
class WavHeader:
    container: str  # RIFF, RF64 or BW64
    sample_type: str  # u8, i16, i24, i32 or f32
    channels: int
    sample_rate: int
    data_offset: int
    data_bytes: int
    frames: int
    file_size: int
    chunks: tuple[str, ...]
    aux_center_hz: int | None = None
    aux_started_at: datetime | None = None  # timezone-aware when the file gives UTC
    aux_recorded_with: str | None = None
    warnings: tuple[str, ...] = field(default=())

    @property
    def sample_width(self) -> int:
        return int(self.sample_type[1:]) // 8

    @property
    def frame_bytes(self) -> int:
        return self.channels * self.sample_width

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate


@dataclass
class _Metadata:
    """Everything collected while walking chunks, before validation."""

    fmt: tuple[int, int, int, int] | None = None  # format, channels, rate, bits
    block_align: int = 0
    ds64_riff: int | None = None
    ds64_data: int | None = None
    aux_center_hz: int | None = None
    aux_started_at: datetime | None = None
    aux_recorded_with: str | None = None
    chunks: list[str] = field(default_factory=list)


def _chunk_name(chunk_id: bytes) -> str:
    return chunk_id.decode("ascii", "replace").strip()


def _read_chunk_header(fh, pos: int) -> tuple[bytes, int] | None:
    fh.seek(pos)
    raw = fh.read(8)
    if len(raw) < 8:
        return None
    return raw[:4], struct.unpack("<I", raw[4:])[0]


def _parse_fmt(body: bytes, meta: _Metadata) -> None:
    if len(body) < 16:
        raise WavError("fmt chunk is too short")
    tag, channels, rate, _byte_rate, align, bits = struct.unpack("<HHIIHH", body[:16])
    if tag == _FORMAT_EXTENSIBLE:
        if len(body) < 26:
            raise WavError("extensible fmt chunk is too short")
        # The first two bytes of the SubFormat GUID hold the real format tag.
        tag = struct.unpack("<H", body[24:26])[0]
    meta.fmt = (tag, channels, rate, bits)
    meta.block_align = align


def _systemtime(raw: bytes) -> datetime | None:
    year, month, _weekday, day, hour, minute, second, ms = struct.unpack("<8H", raw)
    try:
        return datetime(year, month, day, hour, minute, second, ms * 1000)
    except ValueError:
        return None


def _xml_attribute(text: str, name: str) -> str | None:
    match = re.search(rf'\b{name}="([^"]*)"', text)
    return match[1] if match else None


def _parse_auxi_xml(text: str, meta: _Metadata) -> None:
    """SDR Console's form: attributes such as RadioCenterFreq="22002000".

    The attributes are matched directly rather than with an XML parser, because the
    file is untrusted and only a few plain values are needed.
    """
    center = _xml_attribute(text, "RadioCenterFreq")
    if center and center.isdecimal() and int(center) > 0:
        meta.aux_center_hz = int(center)

    seconds = _xml_attribute(text, "UTCSeconds")
    if seconds and seconds.isdecimal():
        try:
            meta.aux_started_at = datetime.fromtimestamp(int(seconds), timezone.utc)
        except (OverflowError, OSError, ValueError):
            pass

    parts = [_xml_attribute(text, "RadioModel"), _xml_attribute(text, "SoftwareName")]
    meta.aux_recorded_with = ", ".join(p for p in parts if p) or None


def _parse_auxi(body: bytes, meta: _Metadata) -> None:
    """The `auxi` chunk, which has two unrelated layouts: XML, or binary fields."""
    if body[:2] == b"\xff\xfe" or body[:2] == b"<\0":
        _parse_auxi_xml(body.decode("utf-16-le", "replace"), meta)
    elif body[:1] == b"<":
        _parse_auxi_xml(body.decode("utf-8", "replace"), meta)
    elif len(body) >= 36:
        # Binary: start time, stop time, then center frequency in Hz. A valid start
        # time is the evidence that this really is the layout we think it is.
        started = _systemtime(body[:16])
        center = struct.unpack("<I", body[32:36])[0]
        if started is not None and center > 0:
            meta.aux_started_at = started
            meta.aux_center_hz = center


def _absorb_chunk(fh, chunk_id: bytes, size: int, body_pos: int, meta: _Metadata) -> None:
    """Record a metadata chunk. Sizes are untrusted, so reads are capped."""
    meta.chunks.append(_chunk_name(chunk_id))
    if chunk_id not in (b"fmt ", b"ds64", b"auxi"):
        return
    fh.seek(body_pos)
    body = fh.read(min(size, _MAX_METADATA_BYTES))
    if chunk_id == b"fmt ":
        _parse_fmt(body, meta)
    elif chunk_id == b"auxi":
        _parse_auxi(body, meta)
    elif len(body) >= 16:
        meta.ds64_riff, meta.ds64_data = struct.unpack("<QQ", body[:16])


def _trailing_chunks_valid(fh, pos: int, file_size: int) -> bool:
    """True if the bytes from `pos` are a clean chain of chunks reaching end of file.

    Sample data almost never passes this: it would need printable ids and sizes
    that land exactly on the next chunk, all the way to the end.
    """
    for _ in range(_MAX_CHUNKS):
        if pos >= file_size - 1:  # a lone final pad byte is fine
            return True
        header = _read_chunk_header(fh, pos)
        if header is None:
            return False
        chunk_id, size = header
        if not all(0x20 <= c <= 0x7E for c in chunk_id):
            return False
        pos += 8 + size + (size & 1)
        if pos > file_size + 1:
            return False
    return False


def _resolve_data_size(
    fh, declared: int, data_offset: int, file_size: int, meta: _Metadata, warnings: list[str]
) -> int:
    """Decide how many bytes of samples exist, trusting the file over its header."""
    available = file_size - data_offset

    if declared == _SIZE_UNKNOWN and meta.ds64_data is not None:
        declared = meta.ds64_data

    if declared in (0, _SIZE_UNKNOWN):
        if available > 0:
            warnings.append(
                f"Header gives no usable data size; using the {available} bytes present in the file."
            )
        return available

    if declared > available:
        warnings.append(
            f"Header claims {declared} data bytes but only {available} exist; the file looks truncated."
        )
        return available

    end = data_offset + declared + (declared & 1)
    if end >= file_size or _trailing_chunks_valid(fh, end, file_size):
        return declared

    warnings.append(
        f"Header claims {declared} data bytes, but {available} follow it and they are not "
        "metadata; the size looks wrong or wrapped at 4 GiB, so using all of them."
    )
    return available


def read_header(path: str | os.PathLike) -> WavHeader:
    """Parse a WAV header. Raises WavError if the file can't be used."""
    with open(path, "rb") as fh:
        file_size = os.fstat(fh.fileno()).st_size
        return _parse(fh, file_size)


def _parse(fh, file_size: int) -> WavHeader:
    head = fh.read(12)
    if len(head) < 12 or head[8:12] != b"WAVE" or head[:4] not in (b"RIFF", b"RF64", b"BW64"):
        raise WavError("not a RIFF, RF64 or BW64 WAVE file")
    container = head[:4].decode("ascii")
    riff_size = struct.unpack("<I", head[4:8])[0]

    meta = _Metadata()
    warnings: list[str] = []
    data_offset = None
    data_bytes = 0

    pos = 12
    for _ in range(_MAX_CHUNKS):
        header = _read_chunk_header(fh, pos)
        if header is None:
            break
        chunk_id, size = header
        if chunk_id == b"data":
            meta.chunks.append("data")
            data_offset = pos + 8
            data_bytes = _resolve_data_size(fh, size, data_offset, file_size, meta, warnings)
            pos = data_offset + data_bytes + (data_bytes & 1)
            continue
        _absorb_chunk(fh, chunk_id, size, pos + 8, meta)
        pos += 8 + size + (size & 1)

    if meta.fmt is None:
        raise WavError("no fmt chunk found")
    if data_offset is None:
        raise WavError("no data chunk found")

    tag, channels, sample_rate, bits = meta.fmt
    sample_type = _SAMPLE_TYPES.get((tag, bits))
    if sample_type is None:
        raise WavError(f"unsupported sample format (tag {tag}, {bits} bits)")
    if channels < 1 or sample_rate < 1:
        raise WavError(f"invalid format: {channels} channels at {sample_rate} Hz")

    frame_bytes = channels * bits // 8
    if meta.block_align != frame_bytes:
        warnings.append(
            f"Header block size is {meta.block_align}, but this format needs {frame_bytes}; using {frame_bytes}."
        )

    extra = data_bytes % frame_bytes
    if extra:
        warnings.append(f"Ignoring {extra} bytes at the end that don't make a whole frame.")
        data_bytes -= extra

    if container != "RIFF" and meta.ds64_data is None:
        warnings.append(f"{container} file has no ds64 chunk; length was taken from the file size.")
    declared_total = meta.ds64_riff if container != "RIFF" and meta.ds64_riff else riff_size
    if declared_total + 8 not in (file_size, file_size - 1) and not any(
        "data" in w for w in warnings
    ):
        warnings.append(f"Header gives a total size of {declared_total + 8}, but the file is {file_size} bytes.")

    return WavHeader(
        container=container,
        sample_type=sample_type,
        channels=channels,
        sample_rate=sample_rate,
        data_offset=data_offset,
        data_bytes=data_bytes,
        frames=data_bytes // frame_bytes,
        file_size=file_size,
        chunks=tuple(meta.chunks),
        aux_center_hz=meta.aux_center_hz,
        aux_started_at=meta.aux_started_at,
        aux_recorded_with=meta.aux_recorded_with,
        warnings=tuple(warnings),
    )


def _decode(raw: bytes, sample_type: str) -> np.ndarray:
    """Convert raw little-endian samples to float32 in the range -1 to just under +1."""
    if sample_type == "u8":  # offset binary: 128 is zero
        return (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_type == "i16":
        return np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if sample_type == "i24":
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        value = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        value = (value ^ 0x800000) - 0x800000  # sign-extend from bit 23
        return (value / 8388608.0).astype(np.float32)
    if sample_type == "i32":
        return (np.frombuffer(raw, "<i4").astype(np.float64) / 2147483648.0).astype(np.float32)
    if sample_type == "f32":
        # Files are untrusted, and one NaN would poison every filter it passes through.
        return np.nan_to_num(np.frombuffer(raw, "<f4"), nan=0.0, posinf=1.0, neginf=-1.0)
    raise WavError(f"unknown sample type {sample_type}")


class WavReader:
    """Reads any range of frames from a recording, without loading the rest."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.header = read_header(path)
        self._fd = os.open(path, os.O_RDONLY)

    def read(self, start: int, count: int) -> np.ndarray:
        """Return up to `count` frames from `start`, shaped (frames, channels).

        Fewer frames come back when the range runs past the end of the recording.
        """
        h = self.header
        if start < 0 or count < 0:
            raise ValueError("start and count must not be negative")
        if count > MAX_READ_FRAMES:
            raise ValueError(f"read of {count} frames exceeds the limit of {MAX_READ_FRAMES}")

        count = max(0, min(count, h.frames - start))
        want = count * h.frame_bytes
        offset = h.data_offset + start * h.frame_bytes
        pieces = []
        while want > 0:  # pread may return less than asked, especially over a network
            piece = os.pread(self._fd, want, offset)
            if not piece:
                break  # the file shrank since its header was read
            pieces.append(piece)
            want -= len(piece)
            offset += len(piece)

        raw = b"".join(pieces)
        raw = raw[: len(raw) - len(raw) % h.frame_bytes]
        return _decode(raw, h.sample_type).reshape(-1, h.channels)

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "WavReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
