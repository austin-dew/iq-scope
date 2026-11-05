"""Run the real app against a temporary folder of synthetic recordings, for browser tests."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import uvicorn

from iqscope.app import create_app
from iqscope.synth import synthesize
from iqscope.wavwrite import write_wav


def main(port: int) -> None:
    # One fixed folder, cleared at each start, so repeated runs can't fill the disk.
    root = Path(tempfile.gettempdir()) / "iqscope-ui-fixture"
    shutil.rmtree(root, ignore_errors=True)
    folder = root / "radio/recordings/laptop"
    folder.mkdir(parents=True)
    (root / "cache").mkdir()

    # A USB signal at +50 kHz, and an AM signal at -100 kHz, in one recording.
    rate = 1_024_000
    usb, _ = synthesize("usb", sample_rate=rate, duration_s=20.0, offset_hz=50_000, message="melody", snr_db=40.0)
    am, _ = synthesize("am", sample_rate=rate, duration_s=20.0, offset_hz=-100_000, message="melody", snr_db=None)
    write_wav(folder / "baseband_7200000Hz_12-00-00_03-11-2026.wav", usb + am, rate)

    audio = 0.4 * np.sin(2 * np.pi * 1000 * np.arange(48_000 * 5) / 48_000)
    write_wav(folder / "audio_7000000Hz_12-00-00_03-11-2026.wav", np.column_stack((audio, audio)), 48_000)

    tone_44k = 0.4 * np.sin(2 * np.pi * 1000 * np.arange(44_100 * 6) / 44_100)
    write_wav(folder / "audio_44100.wav", np.column_stack((tone_44k, tone_44k)), 44_100)
    write_wav(folder / "GigaMono.wav", tone_44k[:44_100], 44_100)

    rng = np.random.default_rng(0)
    write_wav(folder / "mystery.wav", rng.uniform(-0.3, 0.3, (48_000 * 3, 2)), 48_000)
    (folder / "broken.wav").write_bytes(b"this is not a wav file")
    (folder / '<img src=x onerror=alert(1)> "odd".wav').write_bytes(b"neither is this one")

    old = time.time() - 3600
    for path in folder.iterdir():
        os.utime(path, (old, old))

    mountinfo = root / "mountinfo"
    mountinfo.write_text(f"50 22 0:40 / {root / 'radio'} rw - cifs //host/share rw\n")

    app = create_app(str(root / "radio"), str(root / "cache"), str(mountinfo), poll_s=2.0, upload_part_bytes=2**20)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main(int(sys.argv[1]))
