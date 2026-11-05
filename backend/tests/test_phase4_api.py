"""The library, the waterfall tiles and the HTTP API, on real files in a real folder."""

import json
import os
import struct
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from iqscope.app import create_app, encode_frame
from iqscope.library import Library, read_sidecar
from iqscope.recording import open_recording
from iqscope.synth import synthesize
from iqscope.waterfall import (
    BINS,
    DB_MAX,
    DB_MIN,
    PLANNED_TILES_PER_LEVEL,
    TILE_ROWS,
    TileStore,
    compute_tile,
    max_level,
    planned_tiles,
    tile_count,
    tile_seconds,
)
from iqscope.wavwrite import write_wav

RATE = 240_000
OFFSET = 30_000


def write_iq(path, seconds=3.0, offset=OFFSET, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    z, _ = synthesize("am", sample_rate=RATE, duration_s=seconds, offset_hz=offset, snr_db=40.0, **kw)
    write_wav(path, z, RATE, sample_type="f32")
    os.utime(path, (time.time() - 3600, time.time() - 3600))  # long settled


@pytest.fixture
def radio(tmp_path):
    root = tmp_path / "radio"
    write_iq(root / "recordings/laptop/baseband_7200000Hz_12-00-00_03-11-2026.wav")
    return root


@pytest.fixture
def api(tmp_path, radio):
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"50 22 0:40 / {radio} rw - cifs //host/share rw\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    app = create_app(str(radio), str(cache), str(mountinfo), static_dir=None, background=False)
    app.state.library.scan()
    return TestClient(app)


def first_id(api):
    return api.get("/api/recordings").json()["recordings"][0]["id"]


def tile_levels(data):
    return np.frombuffer(data, np.uint8).reshape(TILE_ROWS, BINS)


def to_db(levels):
    return DB_MIN + levels * (DB_MAX - DB_MIN) / 255


def parse_frame(data):
    size = struct.unpack("<I", data[:4])[0]
    assert size % 4 == 0
    header = json.loads(data[4 : 4 + size])
    arrays, pos = {}, 4 + size
    for spec in header["arrays"]:
        arrays[spec["name"]] = np.frombuffer(data, "<f4", spec["length"], pos)
        pos += spec["length"] * 4
    assert pos == len(data)
    return header, arrays


# --- Waterfall math -----------------------------------------------------------------


def test_levels_cover_a_recording_from_seconds_to_hours():
    assert tile_seconds(0) == pytest.approx(5.12)
    assert max_level(1.0) == 0
    assert max_level(5.12) == 0
    assert max_level(60.0) == 4
    assert max_level(12_600.0) == 12
    assert tile_count(60.0, 0) == 12


def test_one_minute_file_gets_full_detail_and_long_file_gets_an_overview():
    short = planned_tiles(60.0)
    assert {level for level, _ in short} == {0, 1, 2, 3, 4}
    assert short[0] == (4, 0)  # coarsest first

    long = planned_tiles(12_600.0)
    assert {level for level, _ in long} == set(range(6, 13))
    assert 0 not in {level for level, _ in long}
    assert sum(1 for level, _ in long if level == min(l for l, _ in long)) <= PLANNED_TILES_PER_LEVEL


def test_tile_shows_a_tone_in_the_right_bin_at_the_right_level(tmp_path):
    path = tmp_path / "baseband_1000000Hz.wav"
    z = 0.25 * np.exp(2j * np.pi * OFFSET * np.arange(RATE * 6) / RATE)
    write_wav(path, z, RATE, sample_type="f32")
    with open_recording(path) as rec:
        levels = tile_levels(compute_tile(rec, 0, 0, 0, 0))
    expected = round((OFFSET + RATE / 2) / RATE * BINS)
    assert np.all(levels[0:250].argmax(axis=1) == expected)
    # A tone of amplitude 0.25 is -12 dBFS.
    assert to_db(levels[10, expected]) == pytest.approx(-12.0, abs=1.5)


def test_tile_time_axis_is_correct(tmp_path):
    """A carrier switched on at exactly 2 s must appear at the row for 2 s."""
    path = tmp_path / "baseband_1000000Hz.wav"
    n = np.arange(RATE * 4)
    z = 0.25 * np.exp(2j * np.pi * OFFSET * n / RATE) * (n >= 2 * RATE)
    write_wav(path, z, RATE, sample_type="f32")
    with open_recording(path) as rec:
        levels = tile_levels(compute_tile(rec, 0, 0, 0, 0))
    bin_index = round((OFFSET + RATE / 2) / RATE * BINS)
    on = levels[:, bin_index] > (0 - DB_MIN) * 255 / (DB_MAX - DB_MIN) * 0.6
    assert int(np.argmax(on)) == pytest.approx(100, abs=1)  # 2 s / 20 ms per row


def test_frequency_zoom_is_sharper_and_finds_the_same_tone(tmp_path):
    path = tmp_path / "baseband_1000000Hz.wav"
    z = 0.25 * np.exp(2j * np.pi * 30_100 * np.arange(RATE * 6) / RATE)
    write_wav(path, z, RATE, sample_type="f32")
    with open_recording(path) as rec:
        fine_x = int((30_100 + RATE / 2) / RATE * 16)
        levels = tile_levels(compute_tile(rec, 0, 4, fine_x, 0))
    binwidth = RATE / (BINS * 16)
    low = -RATE / 2 + fine_x * RATE / 16
    found = low + levels[10].argmax() * binwidth
    assert found == pytest.approx(30_100, abs=binwidth)


def test_rows_past_the_end_are_blank(tmp_path):
    path = tmp_path / "baseband_1000000Hz.wav"
    write_wav(path, 0.25 * np.exp(2j * np.pi * 0.01 * np.arange(RATE)), RATE, sample_type="f32")
    with open_recording(path) as rec:
        levels = tile_levels(compute_tile(rec, 0, 0, 0, 0))
    assert levels[40].max() > 0
    assert levels[51:].max() == 0  # 1 s is 50 rows


def test_coarse_tile_reads_only_a_sample_of_a_long_file(tmp_path, monkeypatch):
    """The whole point: an overview of hours must not read the hours."""
    path = tmp_path / "baseband_1000000Hz.wav"
    fps = RATE * 400
    header = struct.pack("<4sI4s4sIHHIIHH4sQ", b"RIFF", 0, b"WAVE", b"fmt ", 16, 3, 2, RATE, RATE * 8, 8, 32, b"data", 0)[:44]
    with open(path, "wb") as fh:
        fh.write(header)
    os.truncate(path, 44 + fps * 8)
    with open_recording(path, kind_override="iq") as rec:
        assert rec.header.frames == fps
        reads = []
        original = rec.reader.read
        monkeypatch.setattr(rec.reader, "read", lambda s, c: (reads.append(c), original(s, c))[1])
        compute_tile(rec, max_level(rec.header.duration_s), 0, 0, 0)
    assert sum(reads) < fps / 10
    assert len(reads) <= TILE_ROWS


def test_audio_spectrogram_shows_a_tone_at_its_audio_frequency(tmp_path):
    path = tmp_path / "audio_7000000Hz.wav"
    x = 0.5 * np.sin(2 * np.pi * 3_000 * np.arange(48_000 * 2) / 48_000)
    write_wav(path, np.column_stack((x, x)), 48_000)
    with open_recording(path) as rec:
        assert rec.kind == "audio"
        levels = tile_levels(compute_tile(rec, 0, 0, 0, 0))
    expected = round(3_000 / 24_000 * BINS)
    assert levels[10].argmax() == pytest.approx(expected, abs=1)
    assert to_db(levels[10, levels[10].argmax()]) == pytest.approx(-6.0, abs=1.5)


def test_store_trims_oldest_first(tmp_path):
    store = TileStore(tmp_path, max_bytes=2500)
    for t in range(4):
        store.put("k", 0, 0, 0, t, bytes(1000))
        path = store.root / "k" / f"L0_Z0_X0_T{t}.u8"
        os.utime(path, (1000 + t, 1000 + t))
    assert store.trim() == 2
    assert not store.has("k", 0, 0, 0, 0)
    assert not store.has("k", 0, 0, 0, 1)
    assert store.has("k", 0, 0, 0, 3)


def test_store_write_failure_is_not_fatal(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    store = TileStore(blocker)
    store.put("k", 0, 0, 0, 0, b"data")
    assert store.get("k", 0, 0, 0, 0) is None


# --- The watched folder -------------------------------------------------------------


def make_library(root, tmp_path):
    return Library(root / "recordings", TileStore(tmp_path / "cache"))


def test_scan_is_recursive_and_ignores_everything_but_finished_wavs(tmp_path, radio):
    rec = radio / "recordings"
    (rec / "uploads").mkdir()
    (rec / "uploads/incoming.wav.part").write_bytes(b"partial")
    (rec / "notes.txt").write_text("hello")
    (rec / ".hidden.wav").write_bytes(b"x")
    os.symlink("/etc/passwd", rec / "link.wav")
    lib = make_library(radio, tmp_path)
    lib.scan()
    assert [e.path.name for e in lib.list()] == ["baseband_7200000Hz_12-00-00_03-11-2026.wav"]


def test_astronomy_folder_is_not_scanned(tmp_path, radio):
    write_iq(radio / "astronomy/baseband_1420000000Hz.wav")
    lib = make_library(radio, tmp_path)
    lib.scan()
    assert all("astronomy" not in e.relpath for e in lib.list())


def test_new_file_joins_only_after_its_size_stops_changing(tmp_path, radio):
    lib = make_library(radio, tmp_path)
    lib.scan()
    assert len(lib.list()) == 1

    arriving = radio / "recordings/laptop/baseband_9000000Hz.wav"
    now = time.time()
    arriving.write_bytes(b"RIFF" + bytes(100))
    lib.scan(now=now)
    assert len(lib.list()) == 1  # first sighting

    with open(arriving, "ab") as fh:
        fh.write(bytes(100))  # still copying
    lib.scan(now=now + 30)
    assert len(lib.list()) == 1

    lib.scan(now=now + 60)
    assert len(lib.list()) == 2  # unchanged across two scans


def test_removed_file_leaves_the_list(tmp_path, radio):
    lib = make_library(radio, tmp_path)
    lib.scan()
    next(iter(lib.list())).path.unlink()
    lib.scan()
    assert lib.list() == []


def test_share_going_away_keeps_the_list_but_marks_it_unavailable(tmp_path, radio):
    lib = make_library(radio, tmp_path)
    lib.scan()
    away = radio.with_name("radio-away")
    radio.rename(away)
    lib.scan()
    assert lib.online is False
    assert [e.available for e in lib.list()] == [False]
    away.rename(radio)
    lib.scan()
    assert lib.online is True
    assert [e.available for e in lib.list()] == [True]


def test_unreadable_wav_is_listed_with_its_reason_and_others_still_work(tmp_path, radio):
    bad = radio / "recordings/broken.wav"
    bad.write_bytes(b"this is not a wav file at all, not even close")
    os.utime(bad, (time.time() - 3600, time.time() - 3600))
    lib = make_library(radio, tmp_path)
    lib.scan()
    by_name = {e.path.name: e for e in lib.list()}
    assert by_name["broken.wav"].kind == "unreadable"
    assert "RIFF" in by_name["broken.wav"].info["kind_reason"]
    assert len(by_name) == 2


def test_ingest_builds_the_plan_and_survives_a_restart(tmp_path, radio):
    lib = make_library(radio, tmp_path)
    lib.scan()
    entry = lib.list()[0]
    assert entry.tiles_total == len(planned_tiles(entry.info["duration_s"])) >= 1
    assert entry.tiles_done == 0
    for _ in range(entry.tiles_total):
        assert lib.ingest_pending() is True
    assert lib.ingest_pending() is False
    assert entry.tiles_done == entry.tiles_total

    again = make_library(radio, tmp_path)
    again.scan()
    assert again.list()[0].tiles_done == entry.tiles_total  # nothing to rebuild


def test_ingest_skips_recordings_whose_kind_is_unknown(tmp_path, radio):
    path = radio / "recordings/mystery.wav"
    rng = np.random.default_rng(0)
    write_wav(path, rng.uniform(-0.5, 0.5, (48_000, 2)), 48_000)
    os.utime(path, (time.time() - 3600, time.time() - 3600))
    lib = make_library(radio, tmp_path)
    lib.scan()
    entry = next(e for e in lib.list() if e.path.name == "mystery.wav")
    assert entry.kind == "unknown"
    assert entry.tiles_total == 0


def test_sidecar_keeps_only_valid_fields(tmp_path):
    wav = tmp_path / "x.wav"
    sidecar = tmp_path / "x.wav.iqscope.json"
    sidecar.write_text('{"kind":"nonsense","frequency_hz":true,"future":1}')
    assert read_sidecar(wav) == {}
    sidecar.write_text('{"kind":"iq","frequency_hz":7200000}')
    assert read_sidecar(wav) == {"kind": "iq", "frequency_hz": 7200000}
    sidecar.write_text("[not a dict")
    assert read_sidecar(wav) == {}


# --- HTTP: listing ------------------------------------------------------------------


def test_list_describes_a_recording_with_no_paths_exposed(api):
    data = api.get("/api/recordings").json()
    assert data["online"] is True
    item = data["recordings"][0]
    assert item["kind"] == "iq"
    assert item["frequency_hz"] == 7_200_000
    assert item["sample_rate"] == RATE
    assert item["duration_s"] == pytest.approx(3.0)
    assert item["folder"] == "laptop"
    assert item["waterfall"]["low_hz"] == -RATE / 2
    assert "path" not in item and "relpath" not in item
    assert str(api.app.state.library.recordings_dir) not in json.dumps(data)


@pytest.mark.parametrize("bad_id", ["../../../etc/passwd", "..%2f..%2fetc", "nonexistent", "%00"])
def test_ids_cannot_reach_outside_the_library(api, bad_id):
    assert api.get(f"/api/recordings/{bad_id}").status_code in (404, 422)
    assert api.get(f"/api/recordings/{bad_id}/chunk?index=0&offset_hz=0").status_code in (404, 422)


def test_am_default_is_the_bandwidth_chosen_by_ear(api):
    assert api.get("/api/modes").json()["modes"]["am"]["default_bw_hz"] == 6_000


# --- HTTP: chunks -------------------------------------------------------------------


def test_chunk_frame_round_trips():
    data = encode_frame({"a": 1}, {"x": np.array([1.0, 2.0]), "y": np.array([3.0])})
    header, arrays = parse_frame(data)
    assert header["a"] == 1
    assert list(arrays["x"]) == [1.0, 2.0]
    assert list(arrays["y"]) == [3.0]


def test_chunk_returns_real_audio_with_the_right_shape(api):
    rid = first_id(api)
    res = api.get(f"/api/recordings/{rid}/chunk", params={"index": 1, "offset_hz": OFFSET, "mode": "am"})
    assert res.status_code == 200
    header, arrays = parse_frame(res.content)
    assert header["audio_rate"] == 48_000
    assert header["bw_hz"] == 6_000
    assert arrays["audio"].size == 48_000
    assert np.abs(arrays["audio"]).max() > 0.1
    assert set(arrays) == {"audio", "gain_nodes"}  # playback fetches only what it needs


def test_chunk_can_return_the_pairs_that_phase_5_will_show(api):
    rid = first_id(api)
    res = api.get(
        f"/api/recordings/{rid}/chunk",
        params={"index": 1, "offset_hz": OFFSET, "fields": "audio,i,q,carrier,demod,gain_nodes"},
    )
    _, arrays = parse_frame(res.content)
    assert arrays["i"].size == arrays["q"].size == arrays["carrier"].size == 48_000


def test_same_chunk_requested_twice_is_byte_identical(api):
    rid = first_id(api)
    params = {"index": 1, "offset_hz": OFFSET, "mode": "usb"}
    assert api.get(f"/api/recordings/{rid}/chunk", params=params).content == api.get(
        f"/api/recordings/{rid}/chunk", params=params
    ).content


@pytest.mark.parametrize(
    "params, status, phrase",
    [
        ({"index": 0, "offset_hz": 119_000, "mode": "am"}, 422, "only reaches"),
        ({"index": 0, "offset_hz": 0, "mode": "wfm"}, 200, ""),
        ({"index": 0, "offset_hz": 50_000, "mode": "wfm"}, 422, "only reaches"),
        ({"index": 0, "offset_hz": 0, "mode": "am", "bw_hz": 60_000}, 422, "too wide"),
        ({"index": 99, "offset_hz": 0, "mode": "am"}, 404, "outside"),
        ({"index": -1, "offset_hz": 0, "mode": "am"}, 422, ""),
        ({"index": 0, "offset_hz": 0, "mode": "qam"}, 422, ""),
        ({"index": 0, "offset_hz": "abc", "mode": "am"}, 422, ""),
        ({"index": 0, "offset_hz": 0, "bw_hz": -5}, 422, ""),
        ({"index": 0, "offset_hz": 0, "fields": "secret"}, 422, "Unknown field"),
    ],
)
def test_chunk_validates_everything(api, params, status, phrase):
    res = api.get(f"/api/recordings/{first_id(api)}/chunk", params=params)
    assert res.status_code == status
    if phrase:
        assert phrase in str(res.json()["detail"])


def test_read_failure_is_a_clear_error_and_the_app_survives(api):
    rid = first_id(api)
    entry = api.app.state.library.get(rid)
    original = entry.open
    entry.open = lambda: (_ for _ in ()).throw(OSError(112, "Host is down"))
    res = api.get(f"/api/recordings/{rid}/chunk", params={"index": 0, "offset_hz": 0})
    assert res.status_code == 503
    assert "Host is down" in res.json()["detail"]
    entry.open = original
    assert api.get(f"/api/recordings/{rid}/chunk", params={"index": 0, "offset_hz": 0}).status_code == 200


# --- HTTP: waterfall ----------------------------------------------------------------


def test_waterfall_tile_over_http(api):
    rid = first_id(api)
    res = api.get(f"/api/recordings/{rid}/waterfall", params={"level": 0, "t": 0})
    assert res.status_code == 200
    assert len(res.content) == TILE_ROWS * BINS
    levels = tile_levels(res.content)
    expected = round((OFFSET + RATE / 2) / RATE * BINS)
    assert levels[20].argmax() == pytest.approx(expected, abs=1)


@pytest.mark.parametrize(
    "params, status",
    [
        ({"level": 99, "t": 0}, 422),
        ({"level": 5, "t": 0}, 404),
        ({"level": 0, "t": 50}, 404),
        ({"level": 0, "t": -1}, 422),
        ({"level": 0, "t": 0, "z": 9}, 422),
        ({"level": 0, "t": 0, "z": 2, "x": 4}, 404),
    ],
)
def test_waterfall_validates_everything(api, params, status):
    assert api.get(f"/api/recordings/{first_id(api)}/waterfall", params=params).status_code == status


def test_unknown_recording_asks_first_and_then_works(api, radio):
    path = radio / "recordings/mystery.wav"
    rng = np.random.default_rng(0)
    write_wav(path, rng.uniform(-0.5, 0.5, (48_000, 2)), 48_000)
    os.utime(path, (time.time() - 3600, time.time() - 3600))
    lib = api.app.state.library
    lib.scan()
    entry = next(e for e in lib.list() if e.path.name == "mystery.wav")
    assert api.get(f"/api/recordings/{entry.id}/waterfall", params={"level": 0, "t": 0}).status_code == 409

    res = api.put(f"/api/recordings/{entry.id}/sidecar", json={"kind": "iq", "frequency_hz": 7_100_000})
    assert res.status_code == 200
    body = res.json()
    assert body["kind"] == "iq"
    assert body["frequency_hz"] == 7_100_000
    assert body["frequency_source"] == "set by the user"
    assert api.get(f"/api/recordings/{entry.id}/waterfall", params={"level": 0, "t": 0}).status_code == 200
    # The answer lives beside the recording, so it survives a rebuild of the cache.
    assert (path.parent / "mystery.wav.iqscope.json").exists()


def test_sidecar_validation(api):
    rid = first_id(api)
    assert api.put(f"/api/recordings/{rid}/sidecar", json={"frequency_hz": -1}).status_code == 422
    assert api.put(f"/api/recordings/{rid}/sidecar", json={"kind": "video"}).status_code == 422


def test_clearing_a_user_frequency_restores_the_files_own(api):
    rid = first_id(api)
    assert api.put(f"/api/recordings/{rid}/sidecar", json={"frequency_hz": 1_000_000}).json()["frequency_hz"] == 1_000_000
    assert api.put(f"/api/recordings/{rid}/sidecar", json={"frequency_hz": None}).json()["frequency_hz"] == 7_200_000


# --- Security headers and the built page --------------------------------------------


def test_policy_forbids_loading_anything_from_elsewhere(api):
    policy = api.get("/api/health").headers["Content-Security-Policy"]
    assert "default-src 'none'" in policy
    assert "script-src 'self'" in policy
    assert "http" not in policy


def test_built_page_and_audio_worklet_are_served_when_present(tmp_path, radio):
    from pathlib import Path

    import iqscope.app as app_module

    if not Path(app_module._STATIC, "index.html").exists():
        pytest.skip("the frontend has not been built")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("")
    cache = tmp_path / "cache"
    cache.mkdir()
    client = TestClient(create_app(str(radio), str(cache), str(mountinfo), background=False))
    page = client.get("/")
    assert page.status_code == 200
    assert "CONELRAD SIGNAL ANALYZER" in page.text
    assert "http://" not in page.text and "https://" not in page.text
    assert "default-src 'none'" in page.headers["Content-Security-Policy"]
    worklet = client.get("/player-worklet.js")
    assert worklet.status_code == 200
    assert "javascript" in worklet.headers["content-type"]


# --- The close-up spectrum ----------------------------------------------------------


def slice_of(api, tmp_path, z, center, span, index=0, name="baseband_slice.wav"):
    from iqscope.waterfall import compute_slice

    path = tmp_path / name
    write_wav(path, z, RATE, sample_type="f32")
    with open_recording(path) as rec:
        return compute_slice(rec, index, center, span)


def peak_hz_of(header, rows, row):
    return header["first_hz"] + rows[row].argmax() * header["bin_hz"]


def test_slice_finds_a_tone_at_its_exact_frequency_and_level(tmp_path):
    from iqscope.waterfall import compute_slice

    path = tmp_path / "baseband_1000000Hz.wav"
    write_wav(path, 0.25 * np.exp(2j * np.pi * 30_100 * np.arange(RATE * 3) / RATE), RATE, sample_type="f32")
    with open_recording(path) as rec:
        header, rows = compute_slice(rec, 1, 30_000, 5_000)
    assert header["rows"] == 20
    assert abs(peak_hz_of(header, rows, 10) - 30_100) <= header["bin_hz"]
    assert rows[10].max() == pytest.approx(-12.0, abs=1.0)  # amplitude 0.25


def test_slice_edges_and_grid_are_where_they_say(tmp_path):
    from iqscope.waterfall import compute_slice

    path = tmp_path / "baseband_1000000Hz.wav"
    write_wav(path, np.zeros(RATE * 2, dtype=np.complex64), RATE, sample_type="f32")
    with open_recording(path) as rec:
        header, _ = compute_slice(rec, 0, 10_000, 4_000)
    assert header["first_hz"] == pytest.approx(8_000, abs=header["bin_hz"])
    assert header["first_hz"] + header["bins"] * header["bin_hz"] == pytest.approx(12_000, abs=2 * header["bin_hz"])


def test_slice_rows_follow_time(tmp_path):
    """A carrier switched on at 0.5 s appears from the row covering 0.5 s."""
    from iqscope.waterfall import compute_slice

    path = tmp_path / "baseband_1000000Hz.wav"
    n = np.arange(RATE)
    z = 0.25 * np.exp(2j * np.pi * 5_000 * n / RATE) * (n >= RATE // 2)
    write_wav(path, np.concatenate((z, z)), RATE, sample_type="f32")
    with open_recording(path) as rec:
        _, rows = compute_slice(rec, 0, 5_000, 5_000)
    levels = rows.max(axis=1)
    assert levels[7] < -60
    assert levels[12] > -15
    assert int(np.argmax(levels > -30)) in range(8, 11)


def test_wide_slice_keeps_a_narrow_peak_and_stays_small(tmp_path):
    from iqscope.waterfall import SLICE_MAX_BINS, compute_slice

    path = tmp_path / "baseband_1000000Hz.wav"
    write_wav(path, 0.25 * np.exp(2j * np.pi * 40_000 * np.arange(RATE) / RATE), RATE, sample_type="f32")
    with open_recording(path) as rec:
        header, rows = compute_slice(rec, 0, 0, RATE)
    assert header["bins"] <= SLICE_MAX_BINS
    assert abs(peak_hz_of(header, rows, 10) - 40_000) <= header["bin_hz"]
    assert rows[10].max() == pytest.approx(-12.0, abs=1.5)


def test_slice_past_the_band_edge_stays_at_the_floor(tmp_path):
    from iqscope.waterfall import compute_slice

    path = tmp_path / "baseband_1000000Hz.wav"
    write_wav(path, 0.25 * np.exp(2j * np.pi * 100_000 * np.arange(RATE) / RATE), RATE, sample_type="f32")
    with open_recording(path) as rec:
        header, rows = compute_slice(rec, 0, 118_000, 10_000)
    above_edge = int((RATE / 2 - header["first_hz"]) / header["bin_hz"]) + 2
    assert np.all(rows[10, above_edge:] == rows[0, above_edge:])
    assert rows[10, above_edge:].max() <= -139


def test_slice_over_http_validates_and_refuses_audio(api, radio):
    rid = first_id(api)
    ok = api.get(f"/api/recordings/{rid}/slice", params={"index": 1, "center_hz": OFFSET, "span_hz": 10_000})
    assert ok.status_code == 200
    header, arrays = parse_frame(ok.content)
    assert arrays["db"].size == header["rows"] * header["bins"]
    assert len(ok.content) < 100_000
    for params, status in [
        ({"index": 99, "center_hz": 0, "span_hz": 10_000}, 404),
        ({"index": 0, "center_hz": 0, "span_hz": 100}, 422),
        ({"index": 0, "center_hz": 0, "span_hz": RATE * 2}, 422),
        ({"index": 0, "center_hz": RATE, "span_hz": 10_000}, 422),
    ]:
        assert api.get(f"/api/recordings/{rid}/slice", params=params).status_code == status

    path = radio / "recordings/audio_7000000Hz.wav"
    write_wav(path, np.zeros((48_000, 2)), 48_000)
    os.utime(path, (time.time() - 3600, time.time() - 3600))
    api.app.state.library.scan()
    entry = next(e for e in api.app.state.library.list() if e.kind == "audio")
    assert api.get(f"/api/recordings/{entry.id}/slice", params={"index": 0, "center_hz": 0, "span_hz": 10_000}).status_code == 409
