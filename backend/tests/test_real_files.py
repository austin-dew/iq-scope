"""Checks against the user's real recordings. Each test skips if its file isn't present.

Expected values come from inspecting the files directly, including a hex dump of the
first baseband clip, so they don't depend on this code being right.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from iqscope.recording import open_recording

HOME = Path.home()
SDRPP = HOME / ".config/sdrpp/recordings"
SPACE = HOME / "icloudbackup/Space Radio Signals"


def need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path} is not on this machine")
    return path


@pytest.mark.parametrize(
    "name, frames, started",
    [
        ("baseband_9923000Hz_12-58-29_03-11-2026.wav", 25_094_656, datetime(2026, 11, 3, 12, 58, 29)),
        ("baseband_9923000Hz_12-58-58_03-11-2026.wav", 24_870_912, datetime(2026, 11, 3, 12, 58, 58)),
        ("baseband_9923000Hz_12-59-31_03-11-2026.wav", 25_012_224, datetime(2026, 11, 3, 12, 59, 31)),
    ],
)
def test_sdrpp_baseband_clips(name, frames, started):
    with open_recording(need(SDRPP / name)) as rec:
        h = rec.header
        assert rec.kind == "iq"
        assert (h.container, h.sample_type, h.channels) == ("RIFF", "i16", 2)
        assert h.sample_rate == 2_400_000
        assert h.frames == frames
        assert h.data_offset == 44
        assert h.warnings == ()
        assert rec.frequency_hz == 9_923_000
        assert rec.frequency_source == "filename"
        assert rec.started_at == started


def test_first_pair_matches_the_hex_dump():
    """The file starts 9a ea 9a f7: little-endian 0xea9a and 0xf79a."""
    name = "baseband_9923000Hz_12-58-29_03-11-2026.wav"
    with open_recording(need(SDRPP / name)) as rec:
        z = rec.read_iq(0, 1)[0]
    assert z.real == pytest.approx(-5478 / 32768)
    assert z.imag == pytest.approx(-2150 / 32768)


def test_capture_shows_the_rtl_sdr_8_bit_grid():
    name = "baseband_9923000Hz_12-58-29_03-11-2026.wav"
    with open_recording(need(SDRPP / name)) as rec:
        values = np.unique(np.round(rec.reader.read(0, 2_000_000) * 32768).astype(int))
    assert len(values) <= 256
    # Steps of about 256; a level the capture never used makes a multiple of that.
    assert np.diff(values).min() >= 255


@pytest.mark.parametrize(
    "path, rate, channels, lr_match",
    [
        (SDRPP / "audio_4839000Hz_01-18-47_03-11-2026.wav", 48_000, 2, True),
        (HOME / "Desktop/lightning storm.wav", 48_000, 2, None),
        (HOME / "icloudbackup/14085000Hz_morse code beeping.wav", 44_100, 2, None),
        (SPACE / "Preacher 1.wav", 48_000, 2, None),
        (SPACE / "audio_7011100Hz_23-39.wav", 44_100, 2, True),
        (SPACE / "GigaDistortion.wav", 12_000, 1, None),
    ],
)
def test_existing_recordings_are_audio_not_iq(path, rate, channels, lr_match):
    with open_recording(need(path)) as rec:
        assert rec.kind == "audio", rec.kind_reason
        assert rec.header.sample_rate == rate
        assert rec.header.channels == channels
        if lr_match is not None:
            assert rec.lr_match is lr_match
