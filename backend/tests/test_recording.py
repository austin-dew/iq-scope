"""Classification, filename metadata, and the guard against treating audio as IQ."""

from datetime import datetime

import numpy as np
import pytest

from iqscope.recording import (
    NotIQError,
    frequency_from_name,
    open_recording,
    time_from_name,
)
from iqscope.wavwrite import write_wav


def stereo(n=48000, identical=True, seed=0):
    rng = np.random.default_rng(seed)
    left = rng.uniform(-0.5, 0.5, n)
    right = left if identical else rng.uniform(-0.5, 0.5, n)
    return np.column_stack((left, right))


@pytest.mark.parametrize(
    "name, rate, identical, expected",
    [
        ("baseband_9923000Hz_12-58-29_03-11-2026.wav", 2_400_000, False, "iq"),
        ("baseband_100000Hz.wav", 48_000, False, "iq"),  # low-rate IQ, named as such
        ("anything.wav", 1_024_000, False, "iq"),  # rate alone decides, e.g. SDR Console
        ("anything.wav", 250_000, False, "iq"),
        ("anything.wav", 192_000, True, "audio"),  # the audio-rate boundary
        ("audio_4839000Hz_01-18-47_03-11-2026.wav", 48_000, True, "audio"),
        ("audio_7011100Hz_23-39.wav", 44_100, False, "audio"),  # the name decides
        ("lightning storm.wav", 48_000, True, "audio"),  # L = R decides
        ("mystery.wav", 48_000, False, "unknown"),  # could be low-rate IQ or real stereo
        ("mystery.wav", 192_000, False, "unknown"),
    ],
)
def test_two_channel_classification(tmp_path, name, rate, identical, expected):
    path = tmp_path / name
    write_wav(path, stereo(identical=identical), rate)
    with open_recording(path) as rec:
        assert rec.kind == expected, rec.kind_reason


def test_one_channel_is_always_audio(tmp_path):
    path = tmp_path / "baseband_mono.wav"
    write_wav(path, stereo()[:, 0], 2_400_000)
    with open_recording(path) as rec:
        assert rec.kind == "audio"


def test_more_than_two_channels_is_unknown(tmp_path):
    path = tmp_path / "multi.wav"
    write_wav(path, np.zeros((100, 4)), 48_000)
    with open_recording(path) as rec:
        assert rec.kind == "unknown"


def test_small_lr_rounding_still_counts_as_identical(tmp_path):
    frames = stereo()
    frames[:, 1] += 13 / 32768  # the largest difference in the user's real files
    path = tmp_path / "almost.wav"
    write_wav(path, frames, 44_100)
    with open_recording(path) as rec:
        assert rec.kind == "audio"
        assert rec.lr_match


def test_lr_difference_only_late_in_the_file_is_caught(tmp_path):
    frames = stereo(n=100_000)
    frames[90_000:, 1] = -frames[90_000:, 1]
    path = tmp_path / "late.wav"
    write_wav(path, frames, 48_000)
    with open_recording(path) as rec:
        assert rec.kind == "unknown"


# --- Never present audio as IQ ------------------------------------------------------


def test_audio_refuses_read_iq(tmp_path):
    path = tmp_path / "audio_1000Hz.wav"
    write_wav(path, stereo(), 48_000)
    with open_recording(path) as rec:
        with pytest.raises(NotIQError):
            rec.read_iq(0, 10)
        assert rec.read_audio(0, 10).shape == (10, 2)


def test_unknown_refuses_read_iq_until_the_user_answers(tmp_path):
    path = tmp_path / "mystery.wav"
    write_wav(path, stereo(identical=False), 48_000)
    with open_recording(path) as rec:
        with pytest.raises(NotIQError):
            rec.read_iq(0, 10)
    with open_recording(path, kind_override="iq") as rec:
        assert rec.kind_reason == "set by the user"
        assert rec.read_iq(0, 10).dtype == np.complex64


def test_iq_override_needs_two_channels(tmp_path):
    path = tmp_path / "mono.wav"
    write_wav(path, stereo()[:, 0], 48_000)
    with pytest.raises(ValueError, match="two channels"):
        open_recording(path, kind_override="iq")


def test_iq_refuses_read_audio_because_iq_is_not_sound(tmp_path):
    path = tmp_path / "baseband_1000Hz.wav"
    write_wav(path, stereo(identical=False), 2_400_000)
    with open_recording(path) as rec:
        with pytest.raises(ValueError, match="demodulated"):
            rec.read_audio(0, 10)


def test_channel_zero_is_i_and_channel_one_is_q(tmp_path):
    path = tmp_path / "baseband_1000Hz.wav"
    write_wav(path, np.array([[0.5, -0.25]]), 2_400_000)
    with open_recording(path) as rec:
        assert rec.read_iq(0, 1)[0] == pytest.approx(0.5 - 0.25j)


# --- Metadata -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("baseband_9923000Hz_12-58-29_03-11-2026.wav", 9_923_000),
        ("audio_4839000Hz_01-18-47_03-11-2026.wav", 4_839_000),
        ("14085000Hz_morse code beeping.wav", 14_085_000),
        ("14-Mar-2021 123045.500 7.200MHz.wav", 7_200_000),
        ("capture 7200 kHz.wav", 7_200_000),
        ("hydrogen_1.42GHz.wav", 1_420_000_000),
        ("lightning storm.wav", None),
        ("GigaDistortion.wav", None),
        ("rate_48000.wav", None),
        ("0Hz.wav", None),
    ],
)
def test_frequency_from_name(name, expected):
    assert frequency_from_name(name) == expected


def test_time_is_read_day_first():
    assert time_from_name("baseband_9923000Hz_12-58-29_03-11-2026.wav") == datetime(2026, 11, 3, 12, 58, 29)


@pytest.mark.parametrize("name", ["lightning storm.wav", "audio_7011100Hz_23-39.wav", "x_99-99-99_99-99-2026.wav"])
def test_names_without_a_valid_time(name):
    assert time_from_name(name) is None


def test_missing_frequency_is_left_unknown(tmp_path):
    path = tmp_path / "plain.wav"
    write_wav(path, stereo(), 2_400_000)
    with open_recording(path) as rec:
        assert rec.frequency_hz is None
        assert rec.frequency_source is None
