"""Synthetic radio signals with known answers, for testing and the Examples tab.

Each pair is a complex number z = I + jQ: an arrow with a length and an angle.

A plain carrier at `offset_hz` is an arrow of constant length turning steadily:

    z[n] = A * exp(j * 2*pi * offset_hz * n / sample_rate)

Each mode multiplies that carrier by a modulation term built from the message m[n]:

    AM   (1 + 0.8 * m)          arrow length follows the message
    USB  m + j*hilbert(m)       only positive frequencies, the upper sideband
    LSB  the conjugate of USB   only negative frequencies, the lower sideband
    CW   a Morse key, 0 to 1    arrow switches on and off
    FM   exp(j * phase)         spin speed follows the message; length is constant

Signals sit off center so that tuning is required. Before `event_pair` the message is
silent, and it starts there at a known pair, so later phases can prove that pair n
lines up with its audio sample. Anything random comes from a seed, so clips repeat.

FM here has no pre-emphasis: its instantaneous frequency is exactly offset + dev * m.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import hilbert

BANDWIDTH_HZ = {"am": 10_000, "usb": 3_000, "lsb": 3_000, "cw": 500, "nfm": 12_500, "wfm": 200_000}
MODES = tuple(BANDWIDTH_HZ)
AM_DEPTH = 0.8
FM_DEVIATION_HZ = {"nfm": 2_500, "wfm": 75_000}

_MORSE_DIT_S = 0.067  # about 18 words per minute
_MORSE_MESSAGE = (".. ", "--.- ")  # "IQ", then a word gap
_KEY_EDGE_S = 0.005  # soft key edges, as sharp ones would splatter across the band

_MELODY_HZ = (523.25, 659.25, 783.99, 1046.5)
_MELODY_NOTE_S = 0.25


@dataclass(frozen=True)
class Truth:
    """What a clip is meant to contain: the answer key for tests."""

    mode: str
    sample_rate: int
    center_hz: int
    offset_hz: float
    tone_hz: float
    message: str
    event_pair: int
    amplitude: float
    snr_db: float | None
    seed: int

    def as_dict(self) -> dict:
        return asdict(self)


def _message(kind: str, t: np.ndarray, event_s: float, tone_hz: float) -> np.ndarray:
    """The audio to transmit: silent until the event, then within -1 to +1."""
    after = t - event_s
    if kind == "tone":
        m = np.sin(2 * np.pi * tone_hz * after)
    elif kind == "melody":
        notes = np.asarray(_MELODY_HZ)[
            (np.maximum(after, 0) / _MELODY_NOTE_S).astype(int) % len(_MELODY_HZ)
        ]
        # Summing frequency keeps the phase continuous across note changes.
        m = np.sin(2 * np.pi * np.cumsum(notes) * (t[1] - t[0]))
    else:
        raise ValueError(f"unknown message {kind!r}")
    return np.where(after >= 0, m, 0.0)


def _morse_key(t: np.ndarray, event_s: float) -> np.ndarray:
    """A repeating Morse pattern as a 0-to-1 envelope with soft edges."""
    spans, clock = [], 0.0
    for word in _MORSE_MESSAGE:
        for symbol in word:
            if symbol == " ":
                clock += 6 * _MORSE_DIT_S
                continue
            length = (1 if symbol == "." else 3) * _MORSE_DIT_S
            spans.append((clock, clock + length))
            clock += length + _MORSE_DIT_S
    period = clock

    key = np.zeros_like(t)
    after = t - event_s
    repeats = int(np.ceil(max(after[-1], 0) / period)) + 1
    for r in range(repeats):
        for on, off in spans:
            on, off = on + r * period, off + r * period
            rising = np.clip((after - on) / _KEY_EDGE_S, 0, 1)
            falling = np.clip((off - after) / _KEY_EDGE_S, 0, 1)
            key = np.maximum(key, 0.5 - 0.5 * np.cos(np.pi * np.minimum(rising, falling)))
    return key


def _modulation(mode: str, m: np.ndarray, t: np.ndarray, event_s: float, fs: float) -> np.ndarray:
    """The modulation term that multiplies the carrier."""
    if mode == "am":
        return (1 + AM_DEPTH * m).astype(np.complex128)
    if mode == "usb":
        return hilbert(m)
    if mode == "lsb":
        return np.conj(hilbert(m))
    if mode == "cw":
        return _morse_key(t, event_s).astype(np.complex128)
    if mode in FM_DEVIATION_HZ:
        # Phase is the running total of frequency, so the first pair's angle is its
        # own step and every later step is exactly deviation * m[n].
        return np.exp(1j * 2 * np.pi * np.cumsum(FM_DEVIATION_HZ[mode] * m) / fs)
    raise ValueError(f"unknown mode {mode!r}")


def _carrier(n: int, offset_hz: float, fs: float) -> np.ndarray:
    return np.exp(2j * np.pi * offset_hz * np.arange(n) / fs)


def _add_noise(z: np.ndarray, amplitude: float, snr_db: float | None, seed: int) -> np.ndarray:
    """Add complex Gaussian noise, with SNR measured against the carrier amplitude."""
    if snr_db is None:
        return z
    rng = np.random.default_rng(seed)
    sigma = amplitude * 10 ** (-snr_db / 20) / np.sqrt(2)  # half the power in each of I and Q
    return z + sigma * (rng.standard_normal(z.size) + 1j * rng.standard_normal(z.size))


def _check_fits(mode: str, offset_hz: float, fs: float) -> None:
    """Refuse a signal whose edge would fall beyond what the sample rate can represent."""
    edge = abs(offset_hz) + BANDWIDTH_HZ[mode] / 2
    if edge >= fs / 2:
        raise ValueError(f"{mode} at offset {offset_hz} Hz reaches {edge} Hz, beyond +/-{fs / 2} Hz")


def synthesize(
    mode: str,
    *,
    sample_rate: int = 1_024_000,
    duration_s: float = 2.0,
    center_hz: int = 7_200_000,
    offset_hz: float = 50_000.0,
    tone_hz: float = 1_000.0,
    message: str = "tone",
    event_s: float = 0.25,
    amplitude: float = 0.25,
    snr_db: float | None = 40.0,
    seed: int = 1,
) -> tuple[np.ndarray, Truth]:
    """Generate one modulated signal. Returns (pairs as complex64, its Truth)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    _check_fits(mode, offset_hz, sample_rate)

    n = round(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    event_pair = round(event_s * sample_rate)
    exact_event_s = event_pair / sample_rate  # the event falls exactly on a pair

    m = _message(message, t, exact_event_s, tone_hz)
    z = amplitude * _modulation(mode, m, t, exact_event_s, sample_rate)
    z = z * _carrier(n, offset_hz, sample_rate)
    z = _add_noise(z, amplitude, snr_db, seed)

    truth = Truth(
        mode, sample_rate, center_hz, offset_hz, tone_hz, message,
        event_pair, amplitude, snr_db, seed,
    )
    return z.astype(np.complex64), truth


def synthesize_mode_switch(
    *,
    modes: tuple[str, ...] = MODES,
    segment_s: float = 2.0,
    sample_rate: int = 1_024_000,
    center_hz: int = 7_200_000,
    offset_hz: float = 50_000.0,
    tone_hz: float = 1_000.0,
    amplitude: float = 0.25,
    snr_db: float | None = 40.0,
    seed: int = 1,
) -> tuple[np.ndarray, list[dict]]:
    """The same tone through each mode in turn, for comparing them directly.

    Returns the pairs and a list of segments, each with its mode and pair range.
    """
    for mode in modes:
        _check_fits(mode, offset_hz, sample_rate)

    parts, segments = [], []
    start = 0
    for mode in modes:
        z, _ = synthesize(
            mode, sample_rate=sample_rate, duration_s=segment_s, center_hz=center_hz,
            offset_hz=0.0, tone_hz=tone_hz, event_s=0.0, amplitude=amplitude, snr_db=None,
        )
        parts.append(z)
        segments.append({"mode": mode, "start_pair": start, "end_pair": start + z.size})
        start += z.size

    joined = np.concatenate(parts).astype(np.complex128) * _carrier(start, offset_hz, sample_rate)
    return _add_noise(joined, amplitude, snr_db, seed).astype(np.complex64), segments
