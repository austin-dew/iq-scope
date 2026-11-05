"""Markers, frequency marks, example clips and playing audio-only recordings."""

import json
import os
import struct
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from iqscope.app import create_app
from iqscope.dsp import AUDIO_RATE, audio_chunk, chunk_count, demodulate_chunk
from iqscope.examples import EXAMPLES, OFFSET_HZ, SWITCH, ensure_examples, example_info
from iqscope.library import Library, clean_markers, read_sidecar
from iqscope.recording import NotIQError, open_recording
from iqscope.synth import MODES, synthesize
from iqscope.waterfall import TileStore
from iqscope.wavwrite import write_wav

RATE = 240_000


def settle(path):
    os.utime(path, (time.time() - 3600, time.time() - 3600))


@pytest.fixture
def api(tmp_path):
    radio = tmp_path / "radio"
    folder = radio / "recordings"
    folder.mkdir(parents=True)
    z, _ = synthesize("am", sample_rate=RATE, duration_s=3.0, offset_hz=30_000, snr_db=40.0)
    path = folder / "baseband_7200000Hz.wav"
    write_wav(path, z, RATE, sample_type="f32")
    settle(path)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"50 22 0:40 / {radio} rw - cifs //host/share rw\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    app = create_app(str(radio), str(cache), str(mountinfo), static_dir=None, background=False)
    app.state.library.scan()
    client = TestClient(app)
    client.folder = folder
    return client


def first_id(api):
    return api.get("/api/recordings").json()["recordings"][0]["id"]


def parse_frame(data):
    size = struct.unpack("<I", data[:4])[0]
    header = json.loads(data[4 : 4 + size])
    arrays, pos = {}, 4 + size
    for spec in header["arrays"]:
        arrays[spec["name"]] = np.frombuffer(data, "<f4", spec["length"], pos)
        pos += spec["length"] * 4
    return header, arrays


def add_wav(api, name, frames, rate):
    path = api.folder / name
    write_wav(path, frames, rate)
    settle(path)
    api.app.state.library.scan()
    return next(e for e in api.app.state.library.list() if e.path.name == name)


def peak_hz(audio, rate=AUDIO_RATE):
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    return np.fft.rfftfreq(audio.size, 1 / rate)[spectrum.argmax()]


# --- Sidecar answers don't erase each other -----------------------------------------


def test_saving_one_answer_keeps_the_others(api):
    rid = first_id(api)
    sidecar = lambda: api.get(f"/api/recordings/{rid}/sidecar").json()  # noqa: E731
    api.put(f"/api/recordings/{rid}/sidecar", json={"kind": "iq"})
    api.put(f"/api/recordings/{rid}/sidecar", json={"frequency_hz": 7_100_000})
    assert sidecar()["kind"] == "iq"
    assert sidecar()["frequency_hz"] == 7_100_000

    api.put(f"/api/recordings/{rid}/sidecar", json={"markers": [{"t": 1.0, "label": "here"}]})
    assert sidecar()["kind"] == "iq"
    assert sidecar()["frequency_hz"] == 7_100_000

    api.put(f"/api/recordings/{rid}/sidecar", json={"frequency_hz": None})
    assert sidecar()["frequency_hz"] is None
    assert sidecar()["kind"] == "iq"
    assert sidecar()["markers"][0]["label"] == "here"


def test_fields_from_a_newer_version_survive_a_save(api):
    entry = api.app.state.library.list()[0]
    sidecar = entry.path.with_name(entry.path.name + ".iqscope.json")
    sidecar.write_text('{"future_field": 42}')
    api.put(f"/api/recordings/{entry.id}/sidecar", json={"markers": []})
    assert json.loads(sidecar.read_text())["future_field"] == 42


# --- Markers ------------------------------------------------------------------------


def test_markers_save_beside_the_recording_and_return_in_time_order(api):
    rid = first_id(api)
    body = api.put(
        f"/api/recordings/{rid}/sidecar",
        json={"markers": [
            {"id": "b", "t": 2.5, "label": "second"},
            {"id": "a", "t": 0.5, "label": "first"},
            {"id": "c", "t": 1.0, "kind": "frequency", "label": "", "frequency_hz": 20_100_000},
        ]},
    ).json()
    assert [m["id"] for m in body["markers"]] == ["a", "c", "b"]
    assert body["markers"][1]["frequency_hz"] == 20_100_000

    # A new library with an empty cache still finds them: they travel with the recording.
    entry = api.app.state.library.list()[0]
    fresh = Library(api.folder, TileStore(api.folder.parent.parent / "other-cache"))
    fresh.scan()
    assert [m["id"] for m in fresh.get(entry.id).info["markers"]] == ["a", "c", "b"]


def test_markers_edited_on_disk_are_noticed_by_the_next_scan(api):
    entry = api.app.state.library.list()[0]
    sidecar = entry.path.with_name(entry.path.name + ".iqscope.json")
    sidecar.write_text(json.dumps({"markers": [{"t": 1, "label": "by hand"}]}))
    api.app.state.library.scan()
    assert entry.info["markers"][0]["label"] == "by hand"


@pytest.mark.parametrize(
    "markers",
    [
        [{"t": -1}],
        [{"t": "soon"}],
        [{"t": 1, "kind": "banana"}],
        [{"t": 1, "kind": "frequency"}],
        [{"t": 1, "kind": "frequency", "frequency_hz": -5}],
        [{"t": 1, "label": "x" * 201}],
    ],
)
def test_bad_markers_are_refused(api, markers):
    assert api.put(f"/api/recordings/{first_id(api)}/sidecar", json={"markers": markers}).status_code == 422


def test_bad_markers_in_a_sidecar_file_are_dropped_not_fatal():
    raw = [
        {"t": 3, "label": "keep"},
        {"t": True},
        {"t": float("inf")},
        "not a marker",
        {"t": 1, "kind": "frequency", "frequency_hz": True},
        {"t": 2, "label": "<b>" + "y" * 500},
    ]
    clean = clean_markers(raw)
    assert [m["t"] for m in clean] == [2.0, 3.0]
    assert len(clean[0]["label"]) == 200
    assert clean_markers("nonsense") == []


def test_sidecar_with_bad_markers_keeps_its_other_fields(tmp_path):
    wav = tmp_path / "x.wav"
    (tmp_path / "x.wav.iqscope.json").write_text('{"kind": "iq", "markers": "oops"}')
    assert read_sidecar(wav) == {"kind": "iq", "markers": []}


# --- Examples -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    folder = tmp_path_factory.mktemp("examples")
    return folder, ensure_examples(folder)


def test_every_mode_has_an_example_plus_one_that_switches(clips):
    folder, made = clips
    assert len(made) == len(MODES) + 1
    assert {e.mode for e in EXAMPLES} == set(MODES)
    assert (folder / SWITCH.file).exists()
    assert not list(folder.glob("*.tmp")) and not list(folder.glob(".*"))


def test_making_examples_twice_does_nothing(clips):
    assert ensure_examples(clips[0]) == []


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.mode)
def test_each_example_plays_correctly_in_its_own_mode(clips, example):
    with open_recording(clips[0] / example.file) as rec:
        assert rec.kind == "iq"
        assert rec.frequency_hz == 7_200_000
        r = demodulate_chunk(rec, 2, OFFSET_HZ, example.mode)
    assert np.sqrt(np.mean(r.audio.astype(float) ** 2)) > 0.05


def test_ssb_examples_vanish_in_the_wrong_sideband(clips):
    usb = next(e for e in EXAMPLES if e.mode == "usb")
    with open_recording(clips[0] / usb.file) as rec:
        right = demodulate_chunk(rec, 2, OFFSET_HZ, "usb").demod
        wrong = demodulate_chunk(rec, 2, OFFSET_HZ, "lsb").demod
    assert np.sqrt(np.mean(right.astype(float) ** 2)) > 10 * np.sqrt(np.mean(wrong.astype(float) ** 2))


def test_switching_clip_really_changes_mode_at_each_segment(clips):
    info = example_info(SWITCH.file)
    assert [s["mode"] for s in info["segments"]] == list(MODES)
    with open_recording(clips[0] / SWITCH.file) as rec:
        for segment in info["segments"]:
            middle = int((segment["start_s"] + segment["end_s"]) / 2)
            right = demodulate_chunk(rec, middle, OFFSET_HZ, segment["mode"])
            assert np.sqrt(np.mean(right.audio.astype(float) ** 2)) > 0.05, segment["mode"]
    # AM's length varies in its slot; NFM's holds steady in its own.
    with open_recording(clips[0] / SWITCH.file) as rec:
        am = demodulate_chunk(rec, 1, OFFSET_HZ, "am")
        nfm = demodulate_chunk(rec, 9, OFFSET_HZ, "nfm")
    assert np.hypot(am.i, am.q).std() > 5 * np.hypot(nfm.i, nfm.q).std()


def test_examples_appear_last_labeled_and_survive_the_share_going_away(api):
    library = api.app.state.library
    library.prepare_examples()
    data = api.get("/api/recordings").json()["recordings"]
    assert data[0]["example"] is None
    examples = [r for r in data if r["example"]]
    assert len(examples) == len(MODES) + 1
    assert all(r["folder"] == "" for r in examples)
    assert examples[0]["example"]["offset_hz"] == OFFSET_HZ
    assert "path" not in examples[0]

    root = api.folder.parent
    root.rename(root.with_name("radio-away"))
    library.scan()
    after = api.get("/api/recordings").json()
    assert after["online"] is False
    assert [r["available"] for r in after["recordings"] if r["example"]] == [True] * len(examples)
    assert [r["available"] for r in after["recordings"] if not r["example"]] == [False]
    root.with_name("radio-away").rename(root)


def test_a_users_own_folder_named_examples_stays_their_recording(api):
    (api.folder / "examples").mkdir()
    z, _ = synthesize("am", sample_rate=RATE, duration_s=1.0, offset_hz=30_000)
    path = api.folder / "examples/mine.wav"
    write_wav(path, z, RATE)
    settle(path)
    api.app.state.library.scan()
    item = next(r for r in api.get("/api/recordings").json()["recordings"] if r["name"] == "mine.wav")
    assert item["example"] is None
    assert item["folder"] == "examples"


# --- Playing audio-only recordings ----------------------------------------------------


def tone(rate, seconds, hz=1000.0, amplitude=0.5):
    return amplitude * np.sin(2 * np.pi * hz * np.arange(int(rate * seconds)) / rate)


@pytest.mark.parametrize("rate", [48_000, 44_100, 12_000, 96_000])
def test_audio_plays_at_the_right_pitch_and_level_from_any_rate(tmp_path, rate):
    path = tmp_path / "audio_x.wav"
    x = tone(rate, 3.0)
    write_wav(path, np.column_stack((x, x)), rate)
    with open_recording(path) as rec:
        assert rec.kind == "audio"
        out = audio_chunk(rec, 1)
    assert out.dtype == np.float32
    assert out.size == AUDIO_RATE
    assert peak_hz(out) == pytest.approx(1000, abs=1.5)
    assert np.sqrt(np.mean(out.astype(float) ** 2)) == pytest.approx(0.5 / np.sqrt(2), rel=0.02)


@pytest.mark.parametrize("rate", [44_100, 12_000])
def test_audio_chunks_join_without_a_click(tmp_path, rate):
    path = tmp_path / "audio_x.wav"
    x = tone(rate, 4.0)
    write_wav(path, np.column_stack((x, x)), rate)
    with open_recording(path) as rec:
        joined = np.concatenate([audio_chunk(rec, i) for i in (1, 2)]).astype(float)
    ideal_step = 2 * np.pi * 1000 / AUDIO_RATE * 0.5
    assert np.abs(np.diff(joined)).max() < ideal_step * 1.05


def test_audio_position_is_true_after_resampling(tmp_path):
    """A click at exactly 2.5 s must come out at exactly 2.5 s, whatever the source rate."""
    rate = 44_100
    x = np.zeros(rate * 4)
    x[int(2.5 * rate)] = 0.9
    path = tmp_path / "audio_x.wav"
    write_wav(path, np.column_stack((x, x)), rate)
    with open_recording(path) as rec:
        out = audio_chunk(rec, 2)
    assert abs(int(np.abs(out).argmax()) - 24_000) <= 1


def test_mono_and_short_final_chunk(tmp_path):
    path = tmp_path / "x.wav"
    write_wav(path, tone(12_000, 2.5), 12_000)
    with open_recording(path) as rec:
        assert rec.kind == "audio"
        assert chunk_count(rec) == 3
        assert audio_chunk(rec, 2).size == AUDIO_RATE // 2
        with pytest.raises(IndexError):
            audio_chunk(rec, 3)


def test_different_left_and_right_are_mixed_not_dropped(tmp_path):
    rate = 48_000
    path = tmp_path / "audio_x.wav"
    write_wav(path, np.column_stack((tone(rate, 2.0, 1000), tone(rate, 2.0, 3000))), rate)
    with open_recording(path) as rec:
        out = audio_chunk(rec, 0).astype(float)
    spectrum = np.abs(np.fft.rfft(out * np.hanning(out.size)))
    freqs = np.fft.rfftfreq(out.size, 1 / rate)
    for hz in (1000, 3000):
        assert spectrum[np.abs(freqs - hz) < 3].max() > 0.3 * spectrum.max()


def test_audio_chunk_over_http_has_only_sound(api):
    x = tone(44_100, 2.0)
    entry = add_wav(api, "audio_7000000Hz.wav", np.column_stack((x, x)), 44_100)
    res = api.get(
        f"/api/recordings/{entry.id}/chunk",
        params={"index": 1, "offset_hz": 0, "fields": "audio,i,q,demod,gain_nodes,carrier,discriminator"},
    )
    assert res.status_code == 200
    header, arrays = parse_frame(res.content)
    assert set(arrays) == {"audio"}
    assert header["mode"] == "audio"
    assert header["audio_rate"] == AUDIO_RATE
    assert arrays["audio"].size == AUDIO_RATE


def test_audio_is_never_presented_as_iq(api):
    x = tone(48_000, 2.0)
    entry = add_wav(api, "audio_7000000Hz.wav", np.column_stack((x, x)), 48_000)
    with entry.open() as rec:
        with pytest.raises(NotIQError):
            demodulate_chunk(rec, 0, 0, "am")
    assert api.get(f"/api/recordings/{entry.id}/slice", params={"index": 0, "center_hz": 0, "span_hz": 10_000}).status_code == 409


def test_unknown_recording_can_be_answered_as_audio_and_then_plays(api):
    rng = np.random.default_rng(0)
    entry = add_wav(api, "mystery.wav", rng.uniform(-0.3, 0.3, (48_000, 2)), 48_000)
    assert entry.kind == "unknown"
    assert api.get(f"/api/recordings/{entry.id}/chunk", params={"index": 0, "offset_hz": 0}).status_code == 409
    api.put(f"/api/recordings/{entry.id}/sidecar", json={"kind": "audio"})
    assert api.get(f"/api/recordings/{entry.id}/chunk", params={"index": 0, "offset_hz": 0}).status_code == 200


def test_real_audio_recording_plays_if_present():
    from pathlib import Path

    path = Path.home() / "icloudbackup/Space Radio Signals/GigaDistortion.wav"
    if not path.exists():
        pytest.skip(f"{path} is not on this machine")
    with open_recording(path) as rec:
        assert rec.header.sample_rate == 12_000
        out = audio_chunk(rec, 0)
    assert out.size == AUDIO_RATE
    assert np.abs(out).max() > 0
