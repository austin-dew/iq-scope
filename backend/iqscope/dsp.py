"""Turn IQ into sound.

    raw IQ -> shift to 0 Hz -> filter and resample -> mode filter -> demodulate -> gain

Everything is computed for a fixed 1-second chunk, plus padding on both sides that is
discarded afterward. A chunk depends only on its index and settings, never on what was
played before it. That's what lets scrubbing reuse results, and what stops chunks
clicking where they join.

Three rules make pair n line up exactly with its audio sample:
  - Every filter is symmetric and applied centered, so none of them adds delay.
  - Oscillators are computed from the absolute sample number, using exact integer math.
  - Gain depends only on the recording around a position, not on playback history.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, gcd, lcm

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import fftconvolve, firwin, firwin2, kaiserord, resample_poly

from .recording import Recording

AUDIO_RATE = 48_000
CHUNK_SECONDS = 1

# Padding must cover the widest reach inside a chunk: the gain window (350 ms back,
# 150 ms ahead), plus AM's carrier average (100 ms), plus filter tails (under 40 ms).
PRE_ROLL_S = 0.5
POST_ROLL_S = 0.35

SSB_LOW_HZ = 300
CW_TONE_HZ = 700
AM_CARRIER_WINDOW_S = 0.2

# Gain changes smoothly between points 10 ms apart.
GAIN_BLOCK = AUDIO_RATE // 100
AGC_TARGET = 0.5  # the level loud passages are brought to, where 1.0 is full scale
AGC_MAX_GAIN = 1000.0  # how far quiet passages can be raised
_AGC_BACK_BLOCKS = 30
_AGC_AHEAD_BLOCKS = 10
_AGC_SMOOTH_BLOCKS = 5
_FILE_EDGE_GUARD = AUDIO_RATE // 40  # 25 ms, longer than any filter's reach

_STOPBAND_DB = 80
_MAX_WINDOW_PAIRS = 1 << 26
_MAX_TABLE_RATE = 10_000_000  # the table holds 16 bytes per step, so cap its size


class TuningError(ValueError):
    """The requested signal doesn't fit, so the result would be bad audio."""


@dataclass(frozen=True)
class Mode:
    iq_rate: int
    bandwidth_hz: int  # full width, except SSB, where it's the top audio frequency
    fm_deviation_hz: int = 0
    deemphasis_s: float = 0.0


MODES = {
    "am": Mode(48_000, 6_000),  # chosen by ear: 10 kHz let in a neighbor's whine, 5 kHz muffled voices
    "usb": Mode(48_000, 3_000),
    "lsb": Mode(48_000, 3_000),
    "cw": Mode(48_000, 500),
    "nfm": Mode(48_000, 12_500, fm_deviation_hz=2_500, deemphasis_s=530e-6),
    "wfm": Mode(240_000, 200_000, fm_deviation_hz=75_000, deemphasis_s=75e-6),
}


@dataclass(frozen=True)
class ChunkResult:
    """One chunk of audio and the numbers that produced it. PAIR -> SAMPLE shows these."""

    index: int
    mode: str
    source_rate: int
    iq_rate: int
    audio_rate: int
    pairs_per_sample: int  # how many tuned pairs lead to one audio sample
    start_pair: int  # position in the recording, in source pairs
    i: np.ndarray  # tuned pairs after the mode filter, at iq_rate
    q: np.ndarray
    demod: np.ndarray  # demodulated signal before gain, at audio_rate
    gain_nodes: np.ndarray  # gain at every GAIN_BLOCK samples, including the chunk's end
    audio: np.ndarray  # final audio: demod * gain, limited to +/-1
    carrier: np.ndarray | None = None  # AM: the slow average subtracted from the length
    discriminator: np.ndarray | None = None  # FM: frequency from pair to pair, at iq_rate

    def gain_at(self, sample: np.ndarray) -> np.ndarray:
        """The gain applied at the given audio samples of this chunk."""
        return np.interp(sample, np.arange(self.gain_nodes.size) * GAIN_BLOCK, self.gain_nodes)


# --- Filters ------------------------------------------------------------------------


def _lowpass(cutoff_hz: float, transition_hz: float, rate: float) -> np.ndarray:
    """A symmetric low-pass filter with enough taps for its sharpness."""
    numtaps, beta = kaiserord(_STOPBAND_DB, transition_hz / (rate / 2))
    return firwin(numtaps | 1, cutoff_hz, window=("kaiser", beta), fs=rate)


@lru_cache(maxsize=32)
def _resample_filter(source_rate: int, iq_rate: int, half_width_hz: float, up: int) -> np.ndarray:
    """Anti-alias filter for going from source_rate to iq_rate.

    Only energy that would fold back onto the wanted signal has to be removed, so the
    stopband starts at (rate - half width). A narrow signal therefore needs a much
    gentler filter than a brick wall at rate / 2, which saves a lot of computation.
    """
    rate = min(source_rate, iq_rate)
    transition = max(rate - 2 * half_width_hz, 0.1 * rate)
    return _lowpass(rate / 2, transition, source_rate * up)


@lru_cache(maxsize=32)
def _mode_filter(mode: str, bandwidth_hz: float, rate: int) -> np.ndarray:
    """The filter that selects the signal, centered on 0 Hz unless it's SSB."""
    if mode in ("usb", "lsb"):
        # A low-pass filter, slid sideways so its passband covers only one sideband.
        taps = _lowpass((bandwidth_hz - SSB_LOW_HZ) / 2, 200, rate)
        center = (bandwidth_hz + SSB_LOW_HZ) / 2 * (1 if mode == "usb" else -1)
        n = np.arange(taps.size) - taps.size // 2
        return taps * np.exp(2j * np.pi * center * n / rate)
    transition = 100 if mode == "cw" else max(0.1 * bandwidth_hz, 200)
    return _lowpass(bandwidth_hz / 2, transition, rate)


@lru_cache(maxsize=8)
def _deemphasis_filter(tau_s: float, rate: int) -> np.ndarray:
    """FM de-emphasis with no delay.

    The usual one-pole filter delays the audio slightly, which would break the
    pair-to-sample alignment, so this has the same loudness curve but is symmetric.
    """
    corner_hz = 1 / (2 * np.pi * tau_s)
    numtaps = ceil(6 * rate / corner_hz) | 1
    freqs = np.linspace(0, rate / 2, 513)
    return firwin2(numtaps, freqs, 1 / np.sqrt(1 + (freqs / corner_hz) ** 2), fs=rate)


def _filter_centered(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    return fftconvolve(x, taps, mode="same")


@lru_cache(maxsize=4)
def _unit_circle(rate: int) -> np.ndarray:
    """One full turn of the arrow in `rate` steps. Every whole-hertz oscillator at this
    rate only ever lands on these points, so computing them once makes retuning cheap."""
    return np.exp(2j * np.pi * (np.arange(rate) / rate))


def oscillator(start: int, count: int, freq_hz: int, rate: int) -> np.ndarray:
    """exp(j * 2*pi * freq * n / rate) for absolute sample numbers n.

    Multiplying first and reducing modulo the rate keeps this exact in whole numbers.
    Floating point would lose precision hours into a recording, and chunks would click.
    """
    n = np.arange(start, start + count, dtype=np.int64)
    index = (n * freq_hz) % rate
    if rate > _MAX_TABLE_RATE:
        return np.exp(2j * np.pi * (index / rate))
    return _unit_circle(rate)[index]


# --- Checks -------------------------------------------------------------------------


def _signal_half_width(mode: str, bandwidth_hz: float) -> float:
    return bandwidth_hz if mode in ("usb", "lsb") else bandwidth_hz / 2


def check_tuning(source_rate: int, offset_hz: int, mode: str, bandwidth_hz: float) -> None:
    """Refuse anything that can't produce good audio, and say why."""
    spec = MODES[mode]
    half = _signal_half_width(mode, bandwidth_hz)
    if bandwidth_hz <= (SSB_LOW_HZ + 100 if mode in ("usb", "lsb") else 0):
        raise TuningError(f"{mode.upper()} bandwidth of {bandwidth_hz:g} Hz is too narrow.")
    if half >= spec.iq_rate / 2:
        raise TuningError(f"{mode.upper()} bandwidth of {bandwidth_hz:g} Hz is too wide for this mode.")
    if abs(offset_hz) + half > source_rate / 2:
        raise TuningError(
            f"{mode.upper()} needs {half:g} Hz around offset {offset_hz:+d} Hz, but a "
            f"{source_rate:,} S/s recording only reaches +/-{source_rate // 2:,} Hz."
        )
    if half >= source_rate / 2:
        raise TuningError(f"This recording's rate of {source_rate:,} S/s is too low for {mode.upper()}.")


# --- Gain ---------------------------------------------------------------------------


def _gain_nodes(
    signal: np.ndarray, buffer_start: int, count: int, file_start: int, file_end: int
) -> np.ndarray:
    """Gain at each block boundary of [buffer_start, buffer_start + count].

    Gain follows the loudest moment from 300 ms before to 100 ms after each point, so
    it comes down just before a loud sound arrives rather than clipping it. It's
    averaged over +/-50 ms to avoid audible steps. Everything looks only at the
    recording around that point, so the result can't depend on playback history.

    file_start and file_end locate the recording within the buffer. A recording stops
    abruptly at its ends, and filters turn that into a click. The click isn't part of
    the signal, so it mustn't turn the volume down, and the gain fades it out instead.
    """
    usable = signal.size // GAIN_BLOCK
    peaks = np.abs(signal[: usable * GAIN_BLOCK]).reshape(usable, GAIN_BLOCK).max(axis=1)
    starts = np.arange(usable) * GAIN_BLOCK
    peaks[starts < file_start + _FILE_EDGE_GUARD] = 0.0
    peaks[starts + GAIN_BLOCK > file_end - _FILE_EDGE_GUARD] = 0.0

    nodes = ceil(count / GAIN_BLOCK) + 1
    first = buffer_start // GAIN_BLOCK
    reach = _AGC_SMOOTH_BLOCKS
    log_gain = np.empty(nodes + 2 * reach)
    for k in range(log_gain.size):
        center = first - reach + k
        level = peaks[max(center - _AGC_BACK_BLOCKS, 0) : center + _AGC_AHEAD_BLOCKS].max()
        log_gain[k] = np.log(AGC_TARGET / max(level, AGC_TARGET / AGC_MAX_GAIN))
    smoothed = uniform_filter1d(log_gain, size=2 * reach + 1, mode="nearest")
    gain = np.exp(smoothed[reach : reach + nodes])

    # Fade over 20 ms at the very start and end of the recording.
    if buffer_start == file_start:
        gain[0] = 0.0
        gain[1] *= 0.5
    if buffer_start + count == file_end:
        gain[-1] = 0.0
        gain[-2] *= 0.5
    return gain


# --- The pipeline -------------------------------------------------------------------


def _read_padded(rec: Recording, start: int, count: int) -> np.ndarray:
    """Read pairs, with silence beyond the ends of the file so positions stay exact."""
    if count > _MAX_WINDOW_PAIRS:
        raise ValueError(f"window of {count} pairs exceeds the limit of {_MAX_WINDOW_PAIRS}")
    out = np.zeros(count, dtype=np.complex128)
    skip = max(-start, 0)
    pairs = rec.read_iq(max(start, 0), count - skip)
    out[skip : skip + pairs.size] = pairs
    return out


def process_window(
    rec: Recording,
    start_second: int,
    seconds: int,
    offset_hz: int,
    mode: str,
    bandwidth_hz: float | None = None,
    index: int | None = None,
) -> ChunkResult:
    """Demodulate whole seconds of a recording. offset_hz is relative to its center."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if int(offset_hz) != offset_hz:
        raise ValueError("offset_hz must be a whole number of hertz")
    offset_hz = int(offset_hz)
    spec = MODES[mode]
    bandwidth_hz = spec.bandwidth_hz if bandwidth_hz is None else bandwidth_hz
    source_rate = rec.header.sample_rate
    check_tuning(source_rate, offset_hz, mode, bandwidth_hz)

    iq_rate = spec.iq_rate
    pairs_per_sample = iq_rate // AUDIO_RATE
    divisor = gcd(source_rate, iq_rate)
    up, down = iq_rate // divisor, source_rate // divisor

    # Padding has to land exactly on the shared grid of source pairs, tuned pairs,
    # audio samples and gain blocks, or the chunk would be offset by a fraction.
    grid = lcm(up, pairs_per_sample * GAIN_BLOCK)
    pre_iq = ceil(PRE_ROLL_S * iq_rate / grid) * grid
    post_iq = ceil(POST_ROLL_S * iq_rate / grid) * grid
    pre_source = pre_iq * down // up
    post_source = post_iq * down // up

    start_pair = start_second * source_rate
    wanted_pairs = min(seconds * source_rate, rec.header.frames - start_pair)
    if wanted_pairs <= 0:
        raise IndexError(f"second {start_second} is past the end of the recording")

    # Whole seconds of padding space are read even for a short final chunk, so the
    # filters behave identically everywhere; silence fills whatever the file lacks.
    span_pairs = seconds * source_rate
    raw = _read_padded(rec, start_pair - pre_source, pre_source + span_pairs + post_source)

    # 1. Shift the chosen signal to 0 Hz.
    z = raw * oscillator(start_pair - pre_source, raw.size, -offset_hz, source_rate)

    # 2. Filter and resample.
    half = _signal_half_width(mode, bandwidth_hz)
    if up != down:
        taps = _resample_filter(source_rate, iq_rate, half, up)
        z = resample_poly(z, up, down, window=taps)

    # 3. Mode filter.
    z = _filter_centered(z, _mode_filter(mode, bandwidth_hz, iq_rate))

    # Where the recording's first and last pairs fall within this buffer.
    file_start_iq = pre_iq - start_second * iq_rate
    file_end_iq = file_start_iq + rec.header.frames * iq_rate // source_rate

    # 4. Demodulate.
    carrier = discriminator = None
    if mode == "am":
        length = np.abs(z)
        size = round(AM_CARRIER_WINDOW_S * iq_rate) | 1
        carrier = uniform_filter1d(length, size=size, mode="nearest")
        if file_start_iq > 0 or file_end_iq < z.size:
            # Near a file's ends, average only over pairs that exist. Otherwise the
            # silence beyond the end drags the average down and causes a thump.
            exists = np.zeros(z.size)
            exists[max(file_start_iq, 0) : file_end_iq] = 1.0
            carrier = carrier / np.maximum(uniform_filter1d(exists, size=size, mode="nearest"), 1e-3)
        demod = length - carrier
    elif mode in ("usb", "lsb"):
        demod = z.real
    elif mode == "cw":
        # A bare carrier at 0 Hz is silent, so lift it to an audible tone.
        first_iq = start_second * iq_rate - pre_iq
        z = z * oscillator(first_iq, z.size, CW_TONE_HZ, iq_rate)
        demod = z.real
    else:
        # The angle between neighboring pairs is how far the arrow turned: frequency.
        step = np.angle(z[1:] * np.conj(z[:-1]))
        discriminator = np.concatenate(([0.0], step)) * iq_rate / (2 * np.pi * spec.fm_deviation_hz)
        # Beyond the file's ends there's no arrow to measure, so the angle is meaningless
        # noise, and the next filter would smear it into the real audio. Blank it out.
        reach = _mode_filter(mode, bandwidth_hz, iq_rate).size // 2
        discriminator[: max(file_start_iq + reach, 0)] = 0.0
        discriminator[max(file_end_iq - reach, 0) :] = 0.0
        demod = _filter_centered(discriminator, _deemphasis_filter(spec.deemphasis_s, iq_rate))
        if pairs_per_sample > 1:
            taps = _lowpass(AUDIO_RATE * 0.42, AUDIO_RATE * 0.16, iq_rate)
            demod = resample_poly(demod, 1, pairs_per_sample, window=taps)

    # 5. Gain, then cut away the padding.
    pre_audio = pre_iq // pairs_per_sample
    audio_count = wanted_pairs * AUDIO_RATE // source_rate
    iq_count = audio_count * pairs_per_sample
    nodes = _gain_nodes(
        demod, pre_audio, audio_count, file_start_iq // pairs_per_sample, file_end_iq // pairs_per_sample
    )

    demod_out = demod[pre_audio : pre_audio + audio_count]
    gain = np.interp(np.arange(audio_count), np.arange(nodes.size) * GAIN_BLOCK, nodes)
    audio = np.clip(demod_out * gain, -1.0, 1.0)
    iq_slice = slice(pre_iq, pre_iq + iq_count)

    return ChunkResult(
        index=start_second if index is None else index,
        mode=mode,
        source_rate=source_rate,
        iq_rate=iq_rate,
        audio_rate=AUDIO_RATE,
        pairs_per_sample=pairs_per_sample,
        start_pair=start_pair,
        i=z.real[iq_slice].astype(np.float32),
        q=z.imag[iq_slice].astype(np.float32),
        demod=demod_out.astype(np.float32),
        gain_nodes=nodes.astype(np.float32),
        audio=audio.astype(np.float32),
        carrier=None if carrier is None else carrier[iq_slice].astype(np.float32),
        discriminator=None if discriminator is None else discriminator[iq_slice].astype(np.float32),
    )


def audio_chunk(rec: Recording, index: int) -> np.ndarray:
    """One second of an audio recording as mono float32 at AUDIO_RATE.

    A recording that's already sound needs no tuning or demodulation, and no gain either:
    it plays as it was saved. A whole second of padding gives the resampler the same
    surroundings everywhere, so chunks join without a click.
    """
    if index < 0 or index >= chunk_count(rec):
        raise IndexError(f"chunk {index} is outside this recording (0 to {chunk_count(rec) - 1})")
    rate = rec.header.sample_rate
    start = index * rate
    wanted = min(rate, rec.header.frames - start)
    if rate == AUDIO_RATE:
        return np.clip(rec.read_audio(start, wanted).mean(axis=1), -1.0, 1.0).astype(np.float32)

    window = np.zeros(3 * rate)
    first = max(start - rate, 0)
    frames = rec.read_audio(first, min(start + 2 * rate, rec.header.frames) - first)
    window[first - (start - rate) : first - (start - rate) + frames.shape[0]] = frames.mean(axis=1)
    divisor = gcd(rate, AUDIO_RATE)
    resampled = resample_poly(window, AUDIO_RATE // divisor, rate // divisor, window=("kaiser", 12.0))
    count = wanted * AUDIO_RATE // rate
    return np.clip(resampled[AUDIO_RATE : AUDIO_RATE + count], -1.0, 1.0).astype(np.float32)


def chunk_count(rec: Recording) -> int:
    """How many 1-second chunks a recording has, counting a shorter final one."""
    return ceil(rec.header.frames / rec.header.sample_rate / CHUNK_SECONDS)


def demodulate_chunk(
    rec: Recording, index: int, offset_hz: int, mode: str, bandwidth_hz: float | None = None
) -> ChunkResult:
    """Demodulate chunk `index`. The result is the same however this chunk was reached."""
    if index < 0 or index >= chunk_count(rec):
        raise IndexError(f"chunk {index} is outside this recording (0 to {chunk_count(rec) - 1})")
    return process_window(rec, index * CHUNK_SECONDS, CHUNK_SECONDS, offset_hz, mode, bandwidth_hz, index)
