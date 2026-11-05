"""A WAV file plus what it means: IQ or audio, frequency, and start time.

The kind matters most. Two-channel audio and IQ look identical to a WAV header, and
showing audio as IQ would teach something false, so `read_iq` refuses non-IQ files.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .wavfile import WavHeader, WavReader

MAX_AUDIO_RATE = 192_000

# L and R of a demodulated recording can differ by rounding. The largest difference
# seen in the user's files is 13 counts of 32768, about 0.0004.
_NEAR_IDENTICAL = 0.001
_COMPARE_WINDOWS = 8
_COMPARE_FRAMES = 4096

_FREQUENCY = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s?(ghz|mhz|khz|hz)(?![a-z])", re.IGNORECASE)
_UNITS = {"hz": 1, "khz": 1_000, "mhz": 1_000_000, "ghz": 1_000_000_000}
# SDR++ writes the day first: baseband_9923000Hz_12-58-29_03-11-2026.wav is 3 November.
_SDRPP_TIME = re.compile(r"_(\d{2})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{4})(?:\D|$)")


class NotIQError(ValueError):
    """IQ was requested from a recording that isn't known to be IQ."""


def frequencies_in_name(name: str) -> list[int]:
    """Every distinct frequency in a filename, such as 9923000Hz or 7.2MHz, in order."""
    found = []
    for match in _FREQUENCY.finditer(name):
        hz = round(float(match[1]) * _UNITS[match[2].lower()])
        if 0 < hz < 300_000_000_000 and hz not in found:
            found.append(hz)
    return found


def frequency_from_name(name: str) -> int | None:
    """The frequency in a filename, or None if it names none or several.

    With several there's no way to know which one, if any, was the center of the
    recording, and a wrong frequency is worse than an honest "unknown".
    """
    found = frequencies_in_name(name)
    return found[0] if len(found) == 1 else None


def time_from_name(name: str) -> datetime | None:
    """Read the local start time from an SDR++ filename."""
    match = _SDRPP_TIME.search(name)
    if match is None:
        return None
    hour, minute, second, day, month, year = map(int, match.groups())
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def channels_match(reader: WavReader) -> bool:
    """Compare L and R at several points through the file, without reading all of it."""
    h = reader.header
    if h.channels != 2 or h.frames == 0:
        return False
    stride = max(h.frames // _COMPARE_WINDOWS, 1)
    for start in range(0, h.frames, stride):
        frames = reader.read(start, _COMPARE_FRAMES)
        if frames.size and np.abs(frames[:, 0] - frames[:, 1]).max() > _NEAR_IDENTICAL:
            return False
    return True


def classify(name: str, header: WavHeader, lr_match: bool) -> tuple[str, str]:
    """Return (kind, reason), where kind is 'iq', 'audio' or 'unknown'."""
    lower = name.lower()
    if header.channels == 1:
        return "audio", "one channel"
    if header.channels == 2:
        if lower.startswith("baseband"):
            return "iq", "two channels and a baseband filename"
        if header.sample_rate > MAX_AUDIO_RATE:
            return "iq", f"two channels at {header.sample_rate} Hz, above audio rates"
        if lower.startswith("audio_"):
            return "audio", "two channels at an audio rate and an audio_ filename"
        if lr_match:
            return "audio", "two channels at an audio rate, with L and R nearly identical"
        return "unknown", "two different channels at an audio rate: low-rate IQ or real stereo"
    return "unknown", f"{header.channels} channels"


@dataclass
class Recording:
    path: Path
    reader: WavReader
    kind: str
    kind_reason: str
    frequency_hz: int | None
    frequency_source: str | None
    started_at: datetime | None
    lr_match: bool

    @property
    def header(self) -> WavHeader:
        return self.reader.header

    @property
    def frequency_note(self) -> str | None:
        """Why the frequency is unknown, when the filename offers conflicting answers."""
        found = frequencies_in_name(self.path.name)
        if self.frequency_hz is None and len(found) > 1:
            listed = ", ".join(f"{hz:,} Hz" for hz in found)
            return f"Filename names several frequencies ({listed}); the center must be set by hand."
        return None

    def read_iq(self, start: int, count: int) -> np.ndarray:
        """Return pairs as complex64, I + jQ, with channel 0 as I and channel 1 as Q."""
        if self.kind != "iq":
            raise NotIQError(f"{self.path.name} is classified as {self.kind}, not IQ")
        frames = self.reader.read(start, count)
        return frames[:, 0] + 1j * frames[:, 1]

    def read_audio(self, start: int, count: int) -> np.ndarray:
        """Return raw frames of a non-IQ recording, shaped (frames, channels)."""
        if self.kind == "iq":
            raise ValueError(f"{self.path.name} is IQ; it has no sound until demodulated")
        return self.reader.read(start, count)

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> "Recording":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_recording(path: str | os.PathLike, kind_override: str | None = None) -> Recording:
    """Open and classify a recording.

    `kind_override` is the user's own answer, used for files classified as unknown.
    Later phases will store it in the recording's sidecar.
    """
    if kind_override not in (None, "iq", "audio"):
        raise ValueError(f"kind_override must be 'iq' or 'audio', not {kind_override!r}")

    path = Path(path)
    reader = WavReader(path)
    try:
        header = reader.header
        lr_match = channels_match(reader)
        kind, reason = classify(path.name, header, lr_match)
        if kind_override is not None:
            if kind_override == "iq" and header.channels != 2:
                raise ValueError("IQ needs exactly two channels")
            kind, reason = kind_override, "set by the user"

        frequency, source = header.aux_center_hz, "auxi chunk"
        if frequency is None:
            frequency, source = frequency_from_name(path.name), "filename"
        if frequency is None:
            source = None

        return Recording(
            path=path,
            reader=reader,
            kind=kind,
            kind_reason=reason,
            frequency_hz=frequency,
            frequency_source=source,
            started_at=header.aux_started_at or time_from_name(path.name),
            lr_match=lr_match,
        )
    except Exception:
        reader.close()
        raise
