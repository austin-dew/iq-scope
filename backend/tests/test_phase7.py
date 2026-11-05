"""Analysis Mode's raw pairs, and the fonts that ship with the page."""

import os
import struct
import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import iqscope
from iqscope.app import create_app
from iqscope.recording import open_recording
from iqscope.synth import synthesize
from iqscope.wavwrite import write_wav

RATE = 240_000
# In the source tree, and in the package that ends up in the container.
FONT_FOLDERS = [
    Path(__file__).resolve().parents[2] / "frontend" / "public" / "fonts",
    Path(iqscope.__file__).parent / "static" / "fonts",
]


def frame(data: bytes) -> tuple[dict, dict]:
    length = struct.unpack("<I", data[:4])[0]
    header = json.loads(data[4 : 4 + length])
    arrays, at = {}, 4 + length
    for item in header["arrays"]:
        arrays[item["name"]] = np.frombuffer(data, "<f4", item["length"], at)
        at += item["length"] * 4
    return header, arrays


@pytest.fixture
def api(tmp_path):
    radio = tmp_path / "radio"
    folder = radio / "recordings"
    folder.mkdir(parents=True)
    z, _ = synthesize("am", sample_rate=RATE, duration_s=2.0, offset_hz=30_000, snr_db=40.0)
    write_wav(folder / "baseband_7200000Hz.wav", z, RATE, sample_type="f32")
    audio = (np.sin(np.arange(48_000) * 0.05) * 0.5).astype(np.float32)
    write_wav(folder / "sound.wav", audio, 48_000, sample_type="f32")
    for name in os.listdir(folder):
        os.utime(folder / name, (time.time() - 3600, time.time() - 3600))
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"50 22 0:40 / {radio} rw - cifs //host/share rw\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    app = create_app(str(radio), str(cache), str(mountinfo), static_dir=None, background=False)
    app.state.library.scan()
    client = TestClient(app)
    client.path = folder / "baseband_7200000Hz.wav"
    return client


def entry(api, name):
    return next(r for r in api.get("/api/recordings").json()["recordings"] if r["name"] == name)


def test_raw_pairs_are_exactly_what_the_file_holds(api):
    start, count = 12_345, 2_400
    response = api.get(f"/api/recordings/{entry(api, api.path.name)['id']}/raw", params={"start": start, "count": count})
    assert response.status_code == 200
    header, arrays = frame(response.content)
    assert (header["start_pair"], header["count"], header["sample_rate"]) == (start, count, RATE)
    with open_recording(api.path) as rec:
        expected = rec.reader.read(start, count)
    assert np.array_equal(arrays["i"], expected[:, 0].astype("<f4"))
    assert np.array_equal(arrays["q"], expected[:, 1].astype("<f4"))


def test_raw_pairs_stop_at_the_end_of_the_file(api):
    url = f"/api/recordings/{entry(api, api.path.name)['id']}/raw"
    last = 2 * RATE - 10
    header, arrays = frame(api.get(url, params={"start": last, "count": 4096}).content)
    assert header["count"] == 10 and arrays["i"].size == 10
    assert api.get(url, params={"start": 2 * RATE, "count": 1}).status_code == 404


@pytest.mark.parametrize("params", [{"start": 0, "count": 4097}, {"start": 0, "count": 0}, {"start": -1, "count": 10}])
def test_raw_pairs_refuse_requests_outside_the_limits(api, params):
    url = f"/api/recordings/{entry(api, api.path.name)['id']}/raw"
    assert api.get(url, params=params).status_code == 422


def test_an_audio_recording_has_no_raw_pairs(api):
    response = api.get(f"/api/recordings/{entry(api, 'sound.wav')['id']}/raw", params={"start": 0, "count": 10})
    assert response.status_code == 409
    assert "NO IQ DATA" in response.json()["detail"]


@pytest.mark.parametrize("font", ["vt323", "share-tech-mono"])
def test_each_font_ships_with_its_open_license(font):
    folders = [folder for folder in FONT_FOLDERS if folder.is_dir()]
    assert folders
    for folder in folders:
        assert (folder / f"{font}.woff2").read_bytes()[:4] == b"wOF2"
        assert "SIL Open Font License" in (folder / f"LICENSE-{font}.txt").read_text()
