"""Demodulate part of a recording into a plain WAV you can play.

  python -m iqscope.listen FILE --freq 9930000 --mode am --out am.wav
  python -m iqscope.listen FILE --offset 50000 --mode usb --start 2 --seconds 5
"""

from __future__ import annotations

import argparse
import time
from math import ceil, floor
from pathlib import Path

import numpy as np

from .dsp import AUDIO_RATE, MODES, TuningError, chunk_count, demodulate_chunk
from .recording import open_recording
from .wavwrite import write_wav


def render(
    path: str | Path,
    out: str | Path,
    mode: str,
    freq_hz: int | None = None,
    offset_hz: int | None = None,
    start_s: float = 0.0,
    seconds: float | None = None,
    bandwidth_hz: float | None = None,
) -> dict:
    """Write a WAV and return facts about the run. Exactly one of freq or offset is needed."""
    with open_recording(path) as rec:
        if (freq_hz is None) == (offset_hz is None):
            raise ValueError("give exactly one of a frequency or an offset")
        if freq_hz is not None:
            if rec.frequency_hz is None:
                raise ValueError("this recording's center frequency is unknown; use an offset")
            offset_hz = freq_hz - rec.frequency_hz

        end_s = rec.header.duration_s if seconds is None else min(start_s + seconds, rec.header.duration_s)
        first = floor(start_s)
        last = min(ceil(end_s), chunk_count(rec))

        started = time.perf_counter()
        parts = [demodulate_chunk(rec, i, offset_hz, mode, bandwidth_hz).audio for i in range(first, last)]
        elapsed = time.perf_counter() - started

    audio = np.concatenate(parts)
    trim_start = round((start_s - first) * AUDIO_RATE)
    audio = audio[trim_start : trim_start + round((end_s - start_s) * AUDIO_RATE)]
    write_wav(out, audio, AUDIO_RATE)
    return {
        "offset_hz": offset_hz,
        "seconds": audio.size / AUDIO_RATE,
        "elapsed_s": elapsed,
        "chunks": last - first,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--freq", type=int, help="frequency to tune to, in Hz")
    parser.add_argument("--offset", type=int, help="or: distance from the recording's center, in Hz")
    parser.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    parser.add_argument("--seconds", type=float, help="how much to render (default: to the end)")
    parser.add_argument("--bw", type=float, help="bandwidth in Hz; for SSB, the top audio frequency")
    parser.add_argument("--out", help="output file (default: FILE.MODE.wav in this folder)")
    args = parser.parse_args()

    out = args.out or f"{Path(args.file).stem}.{args.mode}.wav"
    try:
        facts = render(args.file, out, args.mode, args.freq, args.offset, args.start, args.seconds, args.bw)
    except (TuningError, ValueError, IndexError) as err:
        raise SystemExit(f"ERROR: {err}")
    per_chunk = facts["elapsed_s"] / max(facts["chunks"], 1)
    print(
        f"{out}: {facts['seconds']:.1f} s at offset {facts['offset_hz']:+d} Hz, "
        f"{per_chunk * 1000:.0f} ms per second of audio"
    )


if __name__ == "__main__":
    main()
