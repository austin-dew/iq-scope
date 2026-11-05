"""Print what IQ Scope sees in a recording: python -m iqscope.info FILE [FILE...]

Read-only. It samples a few spots rather than reading the whole file, so hour-long
recordings on slow media return quickly.
"""

from __future__ import annotations

import sys

import numpy as np

from .recording import open_recording
from .wavfile import WavError

_SPOTS = 16
_SPOT_FRAMES = 65536


def _levels(rec) -> str:
    """RMS and peak from spots spread through the file, in dB relative to full scale."""
    h = rec.header
    if h.frames == 0:
        return "no samples"
    stride = max(h.frames // _SPOTS, 1)
    x = np.concatenate([rec.reader.read(s, _SPOT_FRAMES).ravel() for s in range(0, h.frames, stride)])
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2))
    return (
        f"RMS {20 * np.log10(max(rms, 1e-12)):.1f} dBFS, "
        f"peak {20 * np.log10(max(np.abs(x).max(), 1e-12)):.1f} dBFS, "
        f"{100 * np.mean(np.abs(x) >= 0.99):.3f}% of samples at full scale"
    )


def describe(path: str) -> str:
    with open_recording(path) as rec:
        h = rec.header
        lines = [
            path,
            f"  kind:      {rec.kind.upper()} ({rec.kind_reason})",
            f"  container: {h.container}, chunks: {' '.join(h.chunks)}",
            f"  format:    {h.sample_type}, {h.channels} ch, {h.sample_rate:,} Hz",
            f"  length:    {h.frames:,} frames, {h.duration_s:,.3f} s, {h.file_size / 2**30:.3f} GiB",
            f"  frequency: {f'{rec.frequency_hz:,} Hz from {rec.frequency_source}' if rec.frequency_hz else 'unknown'}",
            f"  started:   {rec.started_at or 'unknown'}",
            f"  levels:    {_levels(rec)}",
        ]
        if h.aux_recorded_with:
            lines.insert(-1, f"  source:    {h.aux_recorded_with}")
        if rec.frequency_note:
            lines.append(f"  NOTE:      {rec.frequency_note}")
        if rec.kind == "iq" and h.frames:
            pairs = "  ".join(f"{z.real:+.4f}{z.imag:+.4f}j" for z in rec.read_iq(0, 3))
            lines.append(f"  pairs:     {pairs}")
        elif h.channels == 2:
            lines.append(f"  L = R:     {'yes (nearly)' if rec.lr_match else 'no'}")
        lines.extend(f"  WARNING:   {w}" for w in h.warnings)
        return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.splitlines()[0], file=sys.stderr)
        return 2
    failed = 0
    for path in argv:
        try:
            print(describe(path))
        except (WavError, OSError) as err:
            print(f"{path}\n  ERROR: {err}")
            failed = 1
    return failed


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
