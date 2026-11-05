"""Demodulator tests. The synthetic clips come with known answers, so nothing is judged
by eye: each mode must recover its tone, at the right pitch, at the right sample."""

import os
from pathlib import Path

import numpy as np
import pytest

from iqscope.dsp import (
    AUDIO_RATE,
    CW_TONE_HZ,
    GAIN_BLOCK,
    TuningError,
    chunk_count,
    demodulate_chunk,
    oscillator,
    process_window,
)
from iqscope.listen import render
from iqscope.recording import NotIQError, open_recording
from iqscope.synth import AM_DEPTH, synthesize
from iqscope.wavwrite import write_wav

RATE = 480_000
OFFSET = 20_000
TONE = 1_000.0
TONE_MODES = ("am", "usb", "lsb", "nfm", "wfm")


def recording(tmp_path, z, rate=RATE, name="baseband_7200000Hz.wav", sample_type="f32"):
    path = tmp_path / name
    write_wav(path, z, rate, sample_type=sample_type)
    return open_recording(path)


@pytest.fixture
def clip(tmp_path):
    """Open a noise-free synthetic clip of one mode as a real file."""

    def make(mode, rate=RATE, offset=OFFSET, duration_s=3.0, event_s=0.25, snr_db=None, **kw):
        z, truth = synthesize(
            mode, sample_rate=rate, duration_s=duration_s, offset_hz=offset,
            event_s=event_s, snr_db=snr_db, **kw,
        )
        return recording(tmp_path, z, rate, f"baseband_{mode}_{rate}_{duration_s}_7200000Hz.wav"), truth

    return make


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def peak_hz(audio):
    window = np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(audio * window))
    return np.fft.rfftfreq(audio.size, 1 / AUDIO_RATE)[spectrum.argmax()]


# --- Each mode recovers its signal --------------------------------------------------


@pytest.mark.parametrize("mode", TONE_MODES)
def test_mode_recovers_the_tone_at_the_right_pitch(clip, mode):
    rec, _ = clip(mode)
    result = demodulate_chunk(rec, 1, OFFSET, mode)
    assert result.audio.size == AUDIO_RATE
    assert peak_hz(result.audio) == pytest.approx(TONE, abs=1.5)
    assert rms(result.audio) > 0.1


@pytest.mark.parametrize("mode", ["am", "lsb", "wfm"])
def test_works_when_the_rate_ratio_is_not_a_whole_number(clip, mode):
    """1.024 MS/s to 48 kS/s is a ratio of 64:3."""
    rec, _ = clip(mode, rate=1_024_000, offset=50_000)
    result = demodulate_chunk(rec, 1, 50_000, mode)
    assert result.audio.size == AUDIO_RATE
    assert peak_hz(result.audio) == pytest.approx(TONE, abs=1.5)


def test_cw_makes_an_audible_tone_only_while_keyed(clip):
    rec, truth = clip("cw")
    result = demodulate_chunk(rec, 1, OFFSET, "cw")
    assert peak_hz(result.audio) == pytest.approx(CW_TONE_HZ, abs=20)
    length = np.hypot(result.i, result.q)
    assert length.max() > 0.5 * truth.amplitude
    assert length.min() < 0.01 * truth.amplitude


def test_voice_style_message_survives_am(clip):
    rec, _ = clip("am", message="melody")
    assert rms(demodulate_chunk(rec, 1, OFFSET, "am").audio) > 0.1


# --- Rejection ----------------------------------------------------------------------


@pytest.mark.parametrize("sent, tuned", [("usb", "lsb"), ("lsb", "usb")])
def test_wrong_sideband_is_rejected(clip, sent, tuned):
    rec, _ = clip(sent)
    wanted = rms(demodulate_chunk(rec, 1, OFFSET, sent).demod)
    unwanted = rms(demodulate_chunk(rec, 1, OFFSET, tuned).demod)
    assert 20 * np.log10(wanted / unwanted) > 60


@pytest.mark.parametrize(
    "neighbor_hz",
    [
        15_000,   # the next channel along
        -15_000,
        48_000,   # would fold exactly onto the tuned signal without filtering
        96_000,
        -144_000,
    ],
)
def test_strong_neighbor_does_not_leak_in(tmp_path, neighbor_hz):
    # Three seconds, so chunk 1 is far from the clicks where the file starts and stops.
    neighbor, truth = synthesize(
        "am", sample_rate=RATE, duration_s=3.0, offset_hz=OFFSET + neighbor_hz, snr_db=None, amplitude=0.25
    )
    wanted, _ = synthesize("am", sample_rate=RATE, duration_s=3.0, offset_hz=OFFSET, snr_db=None, amplitude=0.25)

    with recording(tmp_path, neighbor, name="baseband_n.wav") as rec:
        leak = rms(demodulate_chunk(rec, 1, OFFSET, "am").demod)
    with recording(tmp_path, wanted, name="baseband_w.wav") as rec:
        signal = rms(demodulate_chunk(rec, 1, OFFSET, "am").demod)
    assert 20 * np.log10(signal / max(leak, 1e-12)) > 60


# --- Tuning that can't work is refused ----------------------------------------------


@pytest.mark.parametrize(
    "rate, offset, mode, bandwidth, message",
    [
        (240_000, 50_000, "wfm", None, "only reaches"),
        (192_000, 0, "wfm", None, "only reaches"),
        (480_000, 239_000, "am", None, "only reaches"),
        (480_000, 0, "am", 60_000, "too wide"),
        (480_000, 0, "usb", 200, "too narrow"),
    ],
)
def test_impossible_tuning_is_refused_with_a_reason(tmp_path, rate, offset, mode, bandwidth, message):
    with recording(tmp_path, np.zeros(rate, dtype=np.complex64), rate) as rec:
        with pytest.raises(TuningError, match=message):
            demodulate_chunk(rec, 0, offset, mode, bandwidth)


def test_tune_requires_whole_hertz_and_a_known_mode(clip):
    rec, _ = clip("am")
    with pytest.raises(ValueError, match="whole number"):
        demodulate_chunk(rec, 0, 100.5, "am")
    with pytest.raises(ValueError, match="unknown mode"):
        demodulate_chunk(rec, 0, 0, "qam")


def test_audio_recording_is_refused(tmp_path):
    path = tmp_path / "audio_7000000Hz.wav"
    write_wav(path, np.zeros((48000, 2)), 48_000)
    with open_recording(path) as rec:
        with pytest.raises(NotIQError):
            demodulate_chunk(rec, 0, 0, "am")


# --- Scrubbing: a chunk must not depend on how you got there ------------------------


@pytest.mark.parametrize("mode", ["am", "usb", "lsb", "cw", "nfm", "wfm"])
def test_joined_chunks_match_one_continuous_run(clip, mode):
    rec, _ = clip(mode, duration_s=4.0, snr_db=30.0)
    joined = np.concatenate([demodulate_chunk(rec, i, OFFSET, mode).audio for i in range(4)])
    continuous = process_window(rec, 0, 4, OFFSET, mode).audio
    assert joined.size == continuous.size
    # The difference is what's left of the filters' reach: far below anything audible.
    assert np.abs(joined - continuous).max() < 2e-3


@pytest.mark.parametrize("mode", ["am", "usb", "nfm", "wfm"])
def test_chunk_joins_do_not_click(clip, mode):
    """The step across a join must look like any other step in the audio."""
    rec, _ = clip(mode, duration_s=4.0)
    audio = np.concatenate([demodulate_chunk(rec, i, OFFSET, mode).audio for i in range(1, 4)])
    steps = np.abs(np.diff(audio))
    for boundary in (AUDIO_RATE - 1, 2 * AUDIO_RATE - 1):
        assert steps[boundary] <= steps.max() + 1e-6
    assert steps.max() < 0.3


@pytest.mark.parametrize("mode", ["am", "lsb", "cw", "wfm"])
def test_same_chunk_is_identical_by_playing_or_seeking(clip, mode):
    rec, _ = clip(mode, snr_db=30.0)
    played = [demodulate_chunk(rec, i, OFFSET, mode) for i in range(3)][2]
    seeked = demodulate_chunk(rec, 2, OFFSET, mode)
    for name in ("audio", "i", "q", "demod", "gain_nodes"):
        assert np.array_equal(getattr(played, name), getattr(seeked, name)), name


def test_gain_lines_up_across_a_join(clip):
    rec, _ = clip("am", snr_db=30.0)
    a = demodulate_chunk(rec, 0, OFFSET, "am")
    b = demodulate_chunk(rec, 1, OFFSET, "am")
    assert a.gain_nodes[-1] == pytest.approx(b.gain_nodes[0], rel=0.01)


def test_oscillator_is_exact_hours_into_a_recording():
    """3.5 hours at 2.4 MS/s, where floating-point sample numbers start losing phase."""
    n = 30_240_192_000
    got = oscillator(n, 3, 137_001, 2_400_000)
    exact = np.array([(k * 137_001) % 2_400_000 / 2_400_000 for k in (n, n + 1, n + 2)])
    np.testing.assert_allclose(np.angle(got), np.angle(np.exp(2j * np.pi * exact)), atol=1e-12)


@pytest.mark.parametrize("freq", [137_001, -141_000, 0, 1_199_999])
def test_oscillator_table_matches_direct_calculation(freq):
    start, count, rate = 30_240_192_000, 5000, 2_400_000
    n = np.arange(start, start + count, dtype=np.int64)
    direct = np.exp(2j * np.pi * (((n * freq) % rate) / rate))
    np.testing.assert_array_equal(oscillator(start, count, freq, rate), direct)


def test_oscillator_above_the_table_limit_still_works():
    n = np.arange(100, 110, dtype=np.int64)
    rate = 20_000_000
    np.testing.assert_array_equal(
        oscillator(100, 10, 7, rate), np.exp(2j * np.pi * (((n * 7) % rate) / rate))
    )


def test_tuned_carrier_sits_still_across_a_join(clip):
    """Tuned exactly, a plain carrier is a motionless arrow, even across the boundary."""
    rec, truth = clip("am", duration_s=4.0, event_s=3.9)
    z = [demodulate_chunk(rec, i, OFFSET, "am") for i in (1, 2)]
    pairs = np.concatenate([r.i + 1j * r.q for r in z])
    np.testing.assert_allclose(np.abs(pairs), truth.amplitude, atol=1e-4)
    np.testing.assert_allclose(np.angle(pairs), 0.0, atol=1e-3)


# --- Alignment: the heart of PAIR -> SAMPLE ------------------------------------------


@pytest.mark.parametrize(
    "mode, rate, offset",
    [
        ("am", RATE, OFFSET),
        ("usb", RATE, OFFSET),
        ("lsb", RATE, OFFSET),
        ("nfm", RATE, OFFSET),
        ("wfm", RATE, OFFSET),
        ("am", 1_024_000, 50_000),
        ("usb", 2_400_000, -300_000),
    ],
)
def test_event_at_pair_n_lands_on_the_expected_audio_sample(clip, mode, rate, offset):
    rec, truth = clip(mode, rate=rate, offset=offset, event_s=1.25)
    result = demodulate_chunk(rec, 1, offset, mode)
    expected = (truth.event_pair - result.start_pair) * AUDIO_RATE // rate
    assert expected == 12_000

    # The tone's phase pins the timing down to a fraction of a sample...
    t = (np.arange(result.demod.size) - expected) / AUDIO_RATE
    settled = (t > 0.05) & (t < 0.7)
    x = np.column_stack((np.sin(2 * np.pi * TONE * t[settled]), np.cos(2 * np.pi * TONE * t[settled])))
    sine, cosine = np.linalg.lstsq(x, result.demod[settled].astype(np.float64), rcond=None)[0]
    assert sine > 0
    lag_samples = np.arctan2(cosine, sine) / (2 * np.pi * TONE) * AUDIO_RATE
    assert abs(lag_samples) < 0.5

    # ...and the onset rules out being a whole tone cycle (48 samples) early or late.
    level = np.abs(result.demod)
    onset = int(np.argmax(level > 0.5 * level[settled].max()))
    assert expected <= onset + 12 and onset <= expected + 48


def test_cw_key_down_lands_on_the_expected_sample(clip):
    rec, truth = clip("cw", event_s=1.25, duration_s=3.0)
    result = demodulate_chunk(rec, 1, OFFSET, "cw")
    expected = (truth.event_pair - result.start_pair) * AUDIO_RATE // RATE
    length = np.hypot(result.i, result.q)
    half_up = int(np.argmax(length > truth.amplitude / 2))
    # The key rises over 5 ms (240 samples), so it crosses half at about 120.
    assert half_up == pytest.approx(expected + 120, abs=24)


# --- PAIR -> SAMPLE must show the values the audio actually used ---------------------


def test_am_numbers_reproduce_the_audio(clip):
    rec, _ = clip("am", snr_db=30.0)
    r = demodulate_chunk(rec, 1, OFFSET, "am")
    n = np.arange(r.audio.size)
    computed = (np.hypot(r.i, r.q) - r.carrier) * r.gain_at(n)
    np.testing.assert_allclose(np.clip(computed, -1, 1), r.audio, atol=1e-4)


@pytest.mark.parametrize("mode", ["usb", "lsb", "cw"])
def test_ssb_and_cw_audio_is_i_times_gain(clip, mode):
    rec, _ = clip(mode, snr_db=30.0)
    r = demodulate_chunk(rec, 1, OFFSET, mode)
    computed = r.i * r.gain_at(np.arange(r.audio.size))
    np.testing.assert_allclose(np.clip(computed, -1, 1), r.audio, atol=1e-4)


@pytest.mark.parametrize("mode", ["nfm", "wfm"])
def test_fm_discriminator_comes_from_the_shown_pairs(clip, mode):
    from iqscope.dsp import MODES

    rec, _ = clip(mode, snr_db=30.0)
    r = demodulate_chunk(rec, 1, OFFSET, mode)
    z = r.i.astype(np.float64) + 1j * r.q.astype(np.float64)
    computed = np.angle(z[1:] * np.conj(z[:-1])) * r.iq_rate / (2 * np.pi * MODES[mode].fm_deviation_hz)
    np.testing.assert_allclose(computed, r.discriminator[1:], atol=2e-3)
    np.testing.assert_allclose(r.demod * r.gain_at(np.arange(r.audio.size)), r.audio, atol=1e-4)


def test_wfm_has_five_pairs_per_audio_sample(clip):
    rec, _ = clip("wfm")
    r = demodulate_chunk(rec, 1, OFFSET, "wfm")
    assert r.pairs_per_sample == 5
    assert r.i.size == 5 * r.audio.size == 240_000


def test_fm_recovers_the_known_deviation_before_deemphasis(clip):
    rec, truth = clip("nfm")
    r = demodulate_chunk(rec, 1, OFFSET, "nfm")
    assert r.discriminator.max() == pytest.approx(1.0, abs=0.02)  # m reaches 1 at full deviation


def test_am_demod_level_matches_the_modulation(clip):
    rec, truth = clip("am")
    r = demodulate_chunk(rec, 1, OFFSET, "am")
    assert r.demod.max() == pytest.approx(truth.amplitude * AM_DEPTH, rel=0.02)
    assert r.carrier.mean() == pytest.approx(truth.amplitude, rel=0.01)


# --- Gain ---------------------------------------------------------------------------


def test_gain_brings_weak_and_strong_signals_to_a_similar_level(clip):
    strong, _ = clip("usb", amplitude=0.25)
    weak, _ = clip("usb", amplitude=0.0025, duration_s=3.5)
    loud = rms(demodulate_chunk(strong, 1, OFFSET, "usb").audio)
    quiet = rms(demodulate_chunk(weak, 1, OFFSET, "usb").audio)
    assert 20 * np.log10(loud / quiet) == pytest.approx(0, abs=1)


def test_gain_is_limited_and_silence_stays_silent(tmp_path):
    with recording(tmp_path, np.zeros(RATE * 2, dtype=np.complex64)) as rec:
        r = demodulate_chunk(rec, 0, 0, "am")
    assert np.all(r.audio == 0)
    assert r.gain_nodes.max() <= 1000.0001


def test_gain_nodes_cover_the_whole_chunk(clip):
    rec, _ = clip("am")
    r = demodulate_chunk(rec, 1, OFFSET, "am")
    assert r.gain_nodes.size == AUDIO_RATE // GAIN_BLOCK + 1


def test_audio_never_exceeds_full_scale(clip):
    rec, _ = clip("am", snr_db=20.0)
    for i in range(3):
        assert np.abs(demodulate_chunk(rec, i, OFFSET, "am").audio).max() <= 1.0


# --- The ends of a file -------------------------------------------------------------


@pytest.mark.parametrize("mode", ["nfm", "wfm"])
def test_fm_file_start_has_no_burst_even_when_the_clip_starts_quiet(clip, mode):
    """Quiet means high gain, so any junk from before the file would be very loud."""
    rec, _ = clip(mode, duration_s=3.0, event_s=0.5, snr_db=40.0)
    audio = demodulate_chunk(rec, 0, OFFSET, mode).audio
    assert np.abs(audio[: AUDIO_RATE // 10]).max() < 0.75


def test_short_final_chunk_and_the_end_of_the_file(clip):
    rec, _ = clip("am", duration_s=2.5)
    assert chunk_count(rec) == 3
    assert demodulate_chunk(rec, 2, OFFSET, "am").audio.size == AUDIO_RATE // 2
    with pytest.raises(IndexError):
        demodulate_chunk(rec, 3, OFFSET, "am")
    with pytest.raises(IndexError):
        demodulate_chunk(rec, -1, OFFSET, "am")


def test_first_chunk_works_with_no_audio_before_it(clip):
    rec, _ = clip("am")
    assert demodulate_chunk(rec, 0, OFFSET, "am").audio.size == AUDIO_RATE


@pytest.mark.parametrize("mode", ["am", "usb", "lsb", "nfm", "wfm"])
def test_file_ends_fade_in_and_out_without_a_thump_or_ducked_volume(clip, mode):
    """A strong tone runs from pair 0 to the last pair, then the file just stops."""
    rec, _ = clip(mode, duration_s=3.0, event_s=0.0)
    first = demodulate_chunk(rec, 0, OFFSET, mode)
    middle = demodulate_chunk(rec, 1, OFFSET, mode)
    last = demodulate_chunk(rec, 2, OFFSET, mode)

    assert first.audio[0] == 0
    assert abs(last.audio[-1]) < 0.01
    steady = rms(middle.audio)
    # The sound is at full level almost immediately, not ducked by the click.
    assert rms(first.audio[int(0.03 * AUDIO_RATE) : int(0.2 * AUDIO_RATE)]) == pytest.approx(steady, rel=0.15)
    assert rms(last.audio[-int(0.2 * AUDIO_RATE) : -int(0.03 * AUDIO_RATE)]) == pytest.approx(steady, rel=0.15)
    # And nothing near the ends jumps out above the steady sound.
    assert np.abs(first.audio).max() <= np.abs(middle.audio).max() * 1.2
    assert np.abs(last.audio).max() <= np.abs(middle.audio).max() * 1.2


# --- Rendering a WAV ----------------------------------------------------------------


def test_render_writes_the_requested_slice(clip, tmp_path):
    rec, _ = clip("am")
    out = tmp_path / "out.wav"
    facts = render(rec.path, out, "am", offset_hz=OFFSET, start_s=0.5, seconds=1.25)
    assert facts["seconds"] == pytest.approx(1.25)
    with open_recording(out) as result:
        assert result.kind == "audio"
        assert result.header.sample_rate == AUDIO_RATE
        assert result.header.frames == round(1.25 * AUDIO_RATE)


def test_render_by_absolute_frequency(clip, tmp_path):
    rec, _ = clip("am")
    assert rec.frequency_hz == 7_200_000
    facts = render(rec.path, tmp_path / "o.wav", "am", freq_hz=7_220_000, seconds=1)
    assert facts["offset_hz"] == OFFSET


def test_render_needs_exactly_one_tuning(clip, tmp_path):
    rec, _ = clip("am")
    with pytest.raises(ValueError, match="exactly one"):
        render(rec.path, tmp_path / "o.wav", "am")


# --- The user's real recordings ------------------------------------------------------

SDRPP = Path.home() / ".config/sdrpp/recordings"
VOICE = SDRPP / "baseband_7329000Hz_21-56-25_03-11-2026.wav"
CLIP = SDRPP / "baseband_9923000Hz_12-58-29_03-11-2026.wav"
IO_B = Path(os.environ.get("IQSCOPE_TEST_CARD", "/nonexistent")) / "Jupiter Experiments/Io B/Jupiter_Storm_Watch_02-26-2026_Io_B.wav"


def need(path):
    if not path.exists():
        pytest.skip(f"{path} is not available")
    return path


def test_real_voice_is_on_the_lower_sideband():
    """The energy is just below 7.188 MHz. USB there hears only the steady noise floor,
    while LSB rises 6 to 9 dB above it when the person speaks and drops between phrases."""
    with open_recording(need(VOICE)) as rec:
        offset = 7_188_000 - rec.frequency_hz
        lsb = [rms(demodulate_chunk(rec, i, offset, "lsb").demod) for i in range(1, 9)]
        usb = [rms(demodulate_chunk(rec, i, offset, "usb").demod) for i in range(1, 9)]
    assert 20 * np.log10(np.mean(lsb) / np.mean(usb)) > 5
    assert 20 * np.log10(max(usb) / min(usb)) < 2  # noise holds steady
    assert 20 * np.log10(max(lsb) / min(lsb)) > 5  # speech comes and goes


def test_real_am_station_has_a_strong_steady_carrier():
    with open_recording(need(CLIP)) as rec:
        r = demodulate_chunk(rec, 3, 9_930_000 - rec.frequency_hz, "am")
    assert r.carrier.mean() > 0.05
    assert rms(r.demod) < r.carrier.mean()  # AM's carrier is stronger than its audio
    assert rms(r.audio) > 0.05


def test_real_rf64_chunk_hours_in_matches_itself():
    with open_recording(need(IO_B)) as rec:
        index = chunk_count(rec) - 2
        a = demodulate_chunk(rec, index, -2_000, "usb")
        b = demodulate_chunk(rec, index, -2_000, "usb")
    assert rms(a.demod) > 0
    assert np.array_equal(a.audio, b.audio)
