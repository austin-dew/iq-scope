"""The generator must produce exactly what it claims, since later tests rely on it.

Most checks use snr_db=None, because without noise the math has exact answers.
"""

import numpy as np
import pytest

from iqscope.recording import open_recording
from iqscope.synth import (
    AM_DEPTH,
    FM_DEVIATION_HZ,
    MODES,
    synthesize,
    synthesize_mode_switch,
)
from iqscope.wavwrite import write_wav

RATE = 480_000
OFFSET = 20_000.0
TONE = 1_000.0


def clean(mode, **kw):
    kw = {"sample_rate": RATE, "duration_s": 1.0, "offset_hz": OFFSET, "snr_db": None, **kw}
    return synthesize(mode, **kw)


def spectrum_power(z, fs, freq, half_width=30.0):
    """Power near one frequency, measured with a Hann window."""
    window = np.hanning(z.size)
    power = np.abs(np.fft.fft(z * window)) ** 2 / window.sum() ** 2
    freqs = np.fft.fftfreq(z.size, 1 / fs)
    return power[np.abs(freqs - freq) <= half_width].sum()


def test_am_length_follows_the_message_exactly():
    z, truth = clean("am")
    n = np.arange(z.size)
    m = np.sin(2 * np.pi * TONE * (n - truth.event_pair) / RATE)
    m[: truth.event_pair] = 0
    np.testing.assert_allclose(np.abs(z), truth.amplitude * (1 + AM_DEPTH * m), atol=1e-5)


def test_am_has_a_carrier_and_matching_sidebands():
    z, truth = clean("am")
    after = z[truth.event_pair :]
    carrier = spectrum_power(after, RATE, OFFSET)
    upper = spectrum_power(after, RATE, OFFSET + TONE)
    lower = spectrum_power(after, RATE, OFFSET - TONE)
    assert upper == pytest.approx(lower, rel=0.01)
    # Each sideband has amplitude depth/2 relative to the carrier.
    assert upper / carrier == pytest.approx((AM_DEPTH / 2) ** 2, rel=0.02)


@pytest.mark.parametrize("mode, kept, removed", [("usb", 1, -1), ("lsb", -1, 1)])
def test_ssb_keeps_one_sideband_and_suppresses_the_carrier(mode, kept, removed):
    z, truth = clean(mode)
    after = z[truth.event_pair :]
    wanted = spectrum_power(after, RATE, OFFSET + kept * TONE)
    unwanted = spectrum_power(after, RATE, OFFSET + removed * TONE)
    carrier = spectrum_power(after, RATE, OFFSET)
    assert 10 * np.log10(wanted / unwanted) > 50
    assert 10 * np.log10(wanted / carrier) > 40


@pytest.mark.parametrize("mode", ["nfm", "wfm"])
def test_fm_spin_speed_follows_the_message_and_length_is_constant(mode):
    z, truth = clean(mode)
    np.testing.assert_allclose(np.abs(z), truth.amplitude, atol=1e-5)

    n = np.arange(1, z.size)
    m = np.sin(2 * np.pi * TONE * (n - truth.event_pair) / RATE)
    m[n < truth.event_pair] = 0
    measured = np.angle(z[1:] * np.conj(z[:-1])) * RATE / (2 * np.pi)
    expected = OFFSET + FM_DEVIATION_HZ[mode] * m
    # complex64 limits the precision of angles, hence the tolerance in Hz.
    np.testing.assert_allclose(measured, expected, atol=2.0)


def test_cw_is_silent_until_the_event_then_keys_on_and_off():
    z, truth = clean("cw", duration_s=2.0)
    length = np.abs(z)
    assert length[: truth.event_pair + 1].max() == 0
    # The first dit runs from the event for 67 ms, with 5 ms edges.
    assert length[truth.event_pair + int(0.03 * RATE)] == pytest.approx(truth.amplitude, abs=1e-5)
    # Then a one-dit gap before the second dit.
    assert length[truth.event_pair + int(0.1 * RATE)] == 0
    assert length[truth.event_pair + int(0.17 * RATE)] == pytest.approx(truth.amplitude, abs=1e-5)


@pytest.mark.parametrize("mode", ["am", "nfm", "wfm"])
def test_message_starts_exactly_at_the_event_pair(mode):
    z, truth = clean(mode)
    e = truth.event_pair
    # Before the event, pairs are a plain carrier.
    plain = truth.amplitude * np.exp(2j * np.pi * OFFSET * np.arange(e) / RATE)
    np.testing.assert_allclose(z[:e], plain, atol=1e-5)
    # Soon after it they no longer are.
    n = np.arange(e, e + 200)
    plain_after = truth.amplitude * np.exp(2j * np.pi * OFFSET * n / RATE)
    assert np.abs(z[e:e + 200] - plain_after).max() > 0.01


def test_event_falls_on_a_whole_pair():
    _, truth = clean("am", event_s=0.123456)
    assert truth.event_pair == round(0.123456 * RATE)


def test_melody_stays_within_full_message_range():
    z, truth = clean("am", message="melody")
    assert np.abs(z).max() <= truth.amplitude * (1 + AM_DEPTH) * 1.0001
    assert np.abs(z).min() >= truth.amplitude * (1 - AM_DEPTH) * 0.9999


# --- Noise and repeatability --------------------------------------------------------


def test_same_seed_is_identical_and_different_seed_is_not():
    a, _ = synthesize("am", sample_rate=RATE, duration_s=0.1, seed=7)
    b, _ = synthesize("am", sample_rate=RATE, duration_s=0.1, seed=7)
    c, _ = synthesize("am", sample_rate=RATE, duration_s=0.1, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_noise_level_matches_the_requested_snr():
    with_noise, truth = synthesize("am", sample_rate=RATE, duration_s=1.0, offset_hz=OFFSET, snr_db=20.0, event_s=2.0)
    silent = with_noise - clean("am", event_s=2.0)[0]
    measured = 10 * np.log10(truth.amplitude**2 / np.mean(np.abs(silent) ** 2))
    assert measured == pytest.approx(20.0, abs=0.2)


# --- Limits, mode switching and files -----------------------------------------------


def test_signal_that_does_not_fit_is_refused():
    with pytest.raises(ValueError, match="beyond"):
        synthesize("wfm", sample_rate=240_000, offset_hz=20_000)
    with pytest.raises(ValueError, match="unknown mode"):
        synthesize("qam")


def test_mode_switch_segments_line_up_with_their_mode():
    z, segments = synthesize_mode_switch(sample_rate=RATE, segment_s=0.5, offset_hz=OFFSET, snr_db=None)
    assert [s["mode"] for s in segments] == list(MODES)
    assert segments[0]["start_pair"] == 0
    assert all(a["end_pair"] == b["start_pair"] for a, b in zip(segments, segments[1:]))
    assert segments[-1]["end_pair"] == z.size

    by_mode = {s["mode"]: z[s["start_pair"]:s["end_pair"]] for s in segments}
    assert np.abs(by_mode["am"]).std() > 0.05  # AM varies in length
    assert np.abs(by_mode["nfm"]).std() < 1e-4  # FM doesn't
    assert np.abs(by_mode["cw"]).min() < 1e-4  # CW switches off


@pytest.mark.parametrize("sample_type, tolerance", [("u8", 1 / 128), ("i16", 1 / 32768), ("i32", 1e-6), ("f32", 1e-6)])
@pytest.mark.parametrize("container", ["RIFF", "RF64"])
def test_synthetic_iq_survives_a_trip_through_a_file(tmp_path, sample_type, tolerance, container):
    z, truth = synthesize("usb", sample_rate=RATE, duration_s=0.2, offset_hz=OFFSET)
    path = tmp_path / f"baseband_{truth.center_hz}Hz.wav"
    write_wav(path, z, RATE, sample_type=sample_type, container=container)
    with open_recording(path) as rec:
        assert rec.kind == "iq"
        assert rec.header.container == container
        assert rec.header.warnings == ()
        assert rec.frequency_hz == truth.center_hz
        back = rec.read_iq(0, z.size)
    assert back.size == z.size
    assert np.abs(back.real - z.real).max() <= tolerance
    assert np.abs(back.imag - z.imag).max() <= tolerance


def test_i24_writer_round_trip(tmp_path):
    x = np.array([[-1.0, 0.5], [0.25, -0.25]])
    path = tmp_path / "baseband_24.wav"
    write_wav(path, x, RATE, sample_type="i24")
    with open_recording(path) as rec:
        np.testing.assert_allclose(rec.reader.read(0, 2), x, atol=1e-6)
