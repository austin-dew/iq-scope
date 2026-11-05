#!/usr/bin/env python3
"""Render the Phase 3 listening set into listening/ (not stored in git).

Anything whose source file isn't on this machine is skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

from iqscope.listen import render

ROOT = Path(__file__).parent.parent
OUT = ROOT / "listening"
SDRPP = Path.home() / ".config/sdrpp/recordings"
CLIP = SDRPP / "baseband_9923000Hz_12-58-29_03-11-2026.wav"
VOICE = SDRPP / "baseband_7329000Hz_21-56-25_03-11-2026.wav"
IO_B = Path(os.environ.get("IQSCOPE_TEST_CARD", "/nonexistent")) / "Jupiter Experiments/Io B/Jupiter_Storm_Watch_02-26-2026_Io_B.wav"

JOBS = [
    ("01_voice_7188kHz_LSB", VOICE, dict(mode="lsb", freq_hz=7_188_000)),
    ("02_voice_7188kHz_as_USB_wrong_on_purpose", VOICE, dict(mode="usb", freq_hz=7_188_000)),
    ("03_voice_7188kHz_as_AM_wrong_on_purpose", VOICE, dict(mode="am", freq_hz=7_186_500)),
    ("04_9930kHz_AM", CLIP, dict(mode="am", freq_hz=9_930_000)),
    ("05_9395kHz_AM_weak", CLIP, dict(mode="am", freq_hz=9_395_000)),
    ("06_9832kHz_USB_unidentified", CLIP, dict(mode="usb", freq_hz=9_831_000)),
    ("07_jupiter_IoB_22000kHz_USB_30s", IO_B, dict(mode="usb", freq_hz=22_000_000, start_s=6300, seconds=30)),
]
JOBS += [
    (f"synthetic_{mode}", ROOT / f"examples/example_{mode}_7200000Hz.wav", dict(mode=mode, freq_hz=7_250_000))
    for mode in ("am", "usb", "lsb", "cw", "nfm", "wfm")
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, source, settings in JOBS:
        if not source.exists():
            print(f"SKIPPED {name}: {source} is not available")
            continue
        facts = render(source, OUT / f"{name}.wav", **settings)
        per_second = facts["elapsed_s"] / facts["chunks"] * 1000
        print(f"{name}.wav  {facts['seconds']:.1f} s  ({per_second:.0f} ms per second of audio)")


if __name__ == "__main__":
    main()
