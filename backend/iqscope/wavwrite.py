"""Write IQ or audio to a WAV file in any supported sample type."""

from __future__ import annotations

import os
import struct

import numpy as np

_FORMATS = {"u8": (1, 8), "i16": (1, 16), "i24": (1, 24), "i32": (1, 32), "f32": (3, 32)}
_SIZE_UNKNOWN = 0xFFFFFFFF


def _encode(frames: np.ndarray, sample_type: str) -> bytes:
    """Scale floats in -1..+1 to the file's type, saturating rather than wrapping."""
    x = frames.astype(np.float64).ravel()
    if sample_type == "f32":
        return x.astype("<f4").tobytes()
    if sample_type == "u8":
        return np.clip(np.round(x * 128 + 128), 0, 255).astype(np.uint8).tobytes()
    if sample_type == "i16":
        return np.clip(np.round(x * 32768), -32768, 32767).astype("<i2").tobytes()
    if sample_type == "i32":
        return np.clip(np.round(x * 2147483648), -2147483648, 2147483647).astype("<i4").tobytes()
    if sample_type == "i24":
        v = np.clip(np.round(x * 8388608), -8388608, 8388607).astype("<i4")
        return v.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()  # drop each top byte
    raise ValueError(f"unknown sample type {sample_type!r}")


def write_wav(
    path: str | os.PathLike,
    frames: np.ndarray,
    sample_rate: int,
    sample_type: str = "i16",
    container: str = "RIFF",
) -> None:
    """Write frames to `path`. Complex input is stored as I then Q on two channels."""
    if sample_type not in _FORMATS:
        raise ValueError(f"unknown sample type {sample_type!r}")
    if container not in ("RIFF", "RF64"):
        raise ValueError(f"unknown container {container!r}")

    frames = np.asarray(frames)
    if np.iscomplexobj(frames):
        frames = np.column_stack((frames.real, frames.imag))
    elif frames.ndim == 1:
        frames = frames[:, np.newaxis]
    channels = frames.shape[1]

    tag, bits = _FORMATS[sample_type]
    data = _encode(frames, sample_type)
    align = channels * bits // 8
    fmt = struct.pack("<HHIIHH", tag, channels, sample_rate, sample_rate * align, align, bits)
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    pad = b"\0" if len(data) & 1 else b""

    if container == "RF64":
        ds64_body = struct.pack("<QQQI", 0, len(data), frames.shape[0], 0)
        total = 4 + 8 + len(ds64_body) + len(fmt_chunk) + 8 + len(data) + len(pad)
        ds64_body = struct.pack("<QQQI", total, len(data), frames.shape[0], 0)
        header = (
            b"RF64" + struct.pack("<I", _SIZE_UNKNOWN) + b"WAVE"
            + b"ds64" + struct.pack("<I", len(ds64_body)) + ds64_body
            + fmt_chunk + b"data" + struct.pack("<I", _SIZE_UNKNOWN)
        )
    else:
        total = 4 + len(fmt_chunk) + 8 + len(data) + len(pad)
        header = (
            b"RIFF" + struct.pack("<I", total) + b"WAVE"
            + fmt_chunk + b"data" + struct.pack("<I", len(data))
        )

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(data)
        fh.write(pad)
