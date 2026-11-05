"""Uploads: nothing unfinished or unchecked can ever look like a recording."""

import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from iqscope.app import create_app
from iqscope.synth import synthesize
from iqscope.uploads import STALE_AFTER_S, PartWriter, UploadError, Uploads, piece_list_hash, safe_name
from iqscope.wavwrite import write_wav

PART = 64 * 1024
RATE = 240_000
HEADERS = {"X-IQScope": "1"}
SDRPP_NAME = "baseband_7329000Hz_21-56-25_03-11-2026.wav"


@pytest.fixture
def wav_bytes(tmp_path) -> bytes:
    z, _ = synthesize("am", sample_rate=RATE, duration_s=1.0, offset_hz=30_000, snr_db=40.0)
    path = tmp_path / "source.wav"
    write_wav(path, z, RATE, sample_type="f32")
    return path.read_bytes()


@pytest.fixture
def api(tmp_path):
    radio = tmp_path / "radio"
    (radio / "recordings").mkdir(parents=True)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"50 22 0:40 / {radio} rw - cifs //host/share rw\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    app = create_app(str(radio), str(cache), str(mountinfo), static_dir=None, background=False, upload_part_bytes=PART)
    app.state.library.scan()
    client = TestClient(app)
    client.uploads = radio / "recordings" / "uploads"
    client.mountinfo = mountinfo
    client.make_app = lambda: create_app(
        str(radio), str(cache), str(mountinfo), static_dir=None, background=False, upload_part_bytes=PART
    )
    return client


def pieces(data: bytes, size: int = PART) -> list[bytes]:
    return [data[k : k + size] for k in range(0, len(data), size)]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def start(api, data: bytes, name: str = SDRPP_NAME) -> dict:
    response = api.post("/api/uploads", json={"name": name, "size": len(data)}, headers=HEADERS)
    assert response.status_code == 201, response.text
    return response.json()


def send(api, upload_id: str, offset: int, piece: bytes, digest: str | None = None):
    return api.put(
        f"/api/uploads/{upload_id}/part",
        params={"offset": offset, "sha256": digest or sha(piece)},
        content=piece,
        headers=HEADERS,
    )


def send_all(api, upload_id: str, data: bytes) -> None:
    offset = 0
    for piece in pieces(data):
        assert send(api, upload_id, offset, piece).status_code == 200
        offset += len(piece)


def finish(api, upload_id: str, data: bytes):
    digest = piece_list_hash([sha(p) for p in pieces(data)])
    return api.post(f"/api/uploads/{upload_id}/finish", json={"pieces_sha256": digest}, headers=HEADERS)


def finished_wavs(api) -> list[str]:
    return sorted(p.name for p in api.uploads.glob("*.wav"))


# --- Names --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (SDRPP_NAME, SDRPP_NAME),
        ("../../etc/passwd.wav", "passwd.wav"),
        ("..\\..\\windows\\evil.wav", "evil.wav"),
        ("/etc/cron.d/x.WAV", "x.wav"),
        ('<img src=x onerror=alert(1)> "odd".wav', "img_src_x_onerror_alert_1_odd.wav"),
        (".hidden.wav", "hidden.wav"),
        ("....wav", "upload.wav"),
        ("my recording (1).wav", "my_recording_1.wav"),
        ("a" * 400 + ".wav", "a" * 120 + ".wav"),
    ],
)
def test_names_are_made_safe_and_good_ones_are_left_alone(raw, expected):
    assert safe_name(raw) == expected


@pytest.mark.parametrize("raw", ["x.flac", "x.wav.exe", "wav", "x.part", ""])
def test_only_wav_names_are_allowed(raw):
    with pytest.raises(UploadError) as caught:
        safe_name(raw)
    assert caught.value.status == 415


# --- The normal path ----------------------------------------------------------------------


def test_a_file_arrives_byte_for_byte_and_joins_the_list_straight_away(api, wav_bytes):
    upload = start(api, wav_bytes)
    assert upload["part_bytes"] == PART
    send_all(api, upload["id"], wav_bytes)
    response = finish(api, upload["id"], wav_bytes)
    assert response.status_code == 200
    entry = response.json()
    assert entry["name"] == SDRPP_NAME
    assert entry["kind"] == "iq"
    assert entry["folder"] == "uploads"
    assert entry["frequency_hz"] == 7_329_000  # the name still does its job
    assert (api.uploads / SDRPP_NAME).read_bytes() == wav_bytes
    assert not list(api.uploads.glob("*.part"))
    listed = api.get("/api/recordings").json()["recordings"]
    assert [r["id"] for r in listed] == [entry["id"]]
    assert api.get("/api/uploads").json()["uploads"] == []


def test_a_second_file_with_the_same_name_overwrites_nothing(api, wav_bytes):
    other = wav_bytes[:-4] + b"\x00\x00\x80\x3f"
    for data in (wav_bytes, other, wav_bytes):
        upload = start(api, data)
        send_all(api, upload["id"], data)
        assert finish(api, upload["id"], data).status_code == 200
    stem = SDRPP_NAME[:-4]
    assert finished_wavs(api) == sorted([SDRPP_NAME, f"{stem}-2.wav", f"{stem}-3.wav"])
    assert (api.uploads / SDRPP_NAME).read_bytes() == wav_bytes
    assert (api.uploads / f"{stem}-2.wav").read_bytes() == other


# --- Things going wrong -------------------------------------------------------------------


def test_nothing_is_finished_until_the_end(api, wav_bytes):
    upload = start(api, wav_bytes)
    for offset, piece in zip(range(0, PART * 2, PART), pieces(wav_bytes)[:2]):
        assert send(api, upload["id"], offset, piece).status_code == 200
    api.app.state.library.scan()
    api.app.state.library.scan()
    assert finished_wavs(api) == []
    assert api.get("/api/recordings").json()["recordings"] == []
    response = finish(api, upload["id"], wav_bytes)
    assert response.status_code == 409
    assert response.json()["received"] == PART * 2
    assert finished_wavs(api) == []


def test_a_damaged_piece_is_refused_and_leaves_no_trace(api, wav_bytes):
    upload = start(api, wav_bytes)
    first, second = pieces(wav_bytes)[:2]
    assert send(api, upload["id"], 0, first).status_code == 200
    damaged = bytes([second[0] ^ 1]) + second[1:]
    response = send(api, upload["id"], PART, damaged, digest=sha(second))
    assert response.status_code == 422
    assert response.json()["retry"] is True
    assert (api.uploads / f"{upload['id']}.part").stat().st_size == PART
    assert api.get(f"/api/uploads/{upload['id']}").json()["received"] == PART
    assert send(api, upload["id"], PART, second).status_code == 200


def test_a_short_piece_is_refused(api, wav_bytes):
    upload = start(api, wav_bytes)
    piece = pieces(wav_bytes)[0][:1000]
    assert send(api, upload["id"], 0, piece).status_code == 422
    assert api.get(f"/api/uploads/{upload['id']}").json()["received"] == 0


def test_an_oversized_piece_is_refused(api, wav_bytes):
    upload = start(api, wav_bytes)
    assert send(api, upload["id"], 0, wav_bytes[: PART + 1]).status_code == 413
    assert api.get(f"/api/uploads/{upload['id']}").json()["received"] == 0
    assert (api.uploads / f"{upload['id']}.part").stat().st_size == 0


def test_the_wrong_piece_gets_told_where_to_carry_on(api, wav_bytes):
    upload = start(api, wav_bytes)
    response = send(api, upload["id"], PART, pieces(wav_bytes)[1])
    assert response.status_code == 409
    assert response.json()["received"] == 0


def test_a_mismatched_final_check_refuses_the_file(api, wav_bytes):
    upload = start(api, wav_bytes)
    send_all(api, upload["id"], wav_bytes)
    response = api.post(f"/api/uploads/{upload['id']}/finish", json={"pieces_sha256": "0" * 64}, headers=HEADERS)
    assert response.status_code == 422
    assert finished_wavs(api) == []


def test_something_that_only_claims_to_be_a_wav_is_stopped_at_the_first_piece(api):
    data = b"this is a text file, renamed " * 10_000
    upload = start(api, data, "sneaky.wav")
    response = send(api, upload["id"], 0, pieces(data)[0])
    assert response.status_code == 415
    assert response.json()["retry"] is False


def test_a_wav_with_a_broken_inside_is_stopped_at_the_end(api):
    data = b"RIFF\x00\x00\x00\x00WAVEjunk" + b"\x00" * 200
    upload = start(api, data, "hollow.wav")
    send_all(api, upload["id"], data)
    response = finish(api, upload["id"], data)
    assert response.status_code == 415
    assert finished_wavs(api) == []
    assert not list(api.uploads.glob("*.part"))


def test_an_upload_survives_the_app_restarting(api, wav_bytes):
    upload = start(api, wav_bytes)
    all_pieces = pieces(wav_bytes)
    assert send(api, upload["id"], 0, all_pieces[0]).status_code == 200

    # Bytes written past the record, as a crash mid-piece would leave them.
    with open(api.uploads / f"{upload['id']}.part", "ab") as fh:
        fh.write(b"garbage from a piece that never finished")

    fresh = TestClient(api.make_app())
    state = fresh.get("/api/uploads").json()["uploads"][0]
    assert state["received"] == PART
    assert state["original_name"] == SDRPP_NAME
    assert state["part_hashes"] == [sha(all_pieces[0])]

    offset = PART
    for piece in all_pieces[1:]:
        assert send(fresh, upload["id"], offset, piece).status_code == 200
        offset += len(piece)
    assert finish(fresh, upload["id"], wav_bytes).status_code == 200
    assert (api.uploads / SDRPP_NAME).read_bytes() == wav_bytes


def test_cancelling_removes_everything(api, wav_bytes):
    upload = start(api, wav_bytes)
    assert send(api, upload["id"], 0, pieces(wav_bytes)[0]).status_code == 200
    assert api.delete(f"/api/uploads/{upload['id']}", headers=HEADERS).status_code == 204
    assert list(api.uploads.iterdir()) == []
    assert api.get(f"/api/uploads/{upload['id']}").status_code == 404


def test_a_second_try_waits_for_the_first_to_let_go(tmp_path, wav_bytes):
    uploads = Uploads(tmp_path / "up", tmp_path / "state", PART)
    (tmp_path / "up").mkdir()
    state = uploads.create("a.wav", len(wav_bytes))
    digest = sha(pieces(wav_bytes)[0])
    writer = uploads.begin_part(state["id"], 0, digest)
    with pytest.raises(UploadError) as caught:
        uploads.begin_part(state["id"], 0, digest)
    assert caught.value.status == 409 and caught.value.retry
    writer.abandon()
    writer.close()
    uploads.begin_part(state["id"], 0, digest).close()


def test_abandoning_a_piece_cuts_it_off(tmp_path, wav_bytes):
    uploads = Uploads(tmp_path / "up", tmp_path / "state", PART)
    (tmp_path / "up").mkdir()
    state = uploads.create("a.wav", len(wav_bytes))
    writer: PartWriter = uploads.begin_part(state["id"], 0, sha(pieces(wav_bytes)[0]))
    writer.write(wav_bytes[:1000])
    writer.abandon()
    writer.close()
    assert uploads.part_path(state["id"]).stat().st_size == 0
    assert uploads.get(state["id"])["received"] == 0


def test_abandoned_uploads_are_cleared_after_two_weeks_and_not_before(tmp_path, wav_bytes):
    uploads = Uploads(tmp_path / "up", tmp_path / "state", PART)
    (tmp_path / "up").mkdir()
    state = uploads.create("a.wav", len(wav_bytes))
    uploads.purge_stale(now=state["updated_at"] + STALE_AFTER_S - 60)
    assert uploads.part_path(state["id"]).exists()
    uploads.purge_stale(now=state["updated_at"] + STALE_AFTER_S + 60)
    assert not uploads.part_path(state["id"]).exists()
    assert uploads.unfinished() == []


def test_no_upload_starts_without_room(api, wav_bytes, monkeypatch):
    import shutil as real

    class Usage:
        free = len(wav_bytes)

    monkeypatch.setattr(real, "disk_usage", lambda _path: Usage)
    response = api.post("/api/uploads", json={"name": "a.wav", "size": len(wav_bytes)}, headers=HEADERS)
    assert response.status_code == 507
    assert "Not enough room" in response.json()["detail"]


def test_no_upload_starts_while_the_archive_is_offline(api, tmp_path):
    api.mountinfo.write_text(f"50 22 0:40 / {tmp_path}/radio rw - ext4 /dev/nvme0n1p2 rw\n")
    response = api.post("/api/uploads", json={"name": "a.wav", "size": 1000}, headers=HEADERS)
    assert response.status_code == 503
    assert "offline" in response.json()["detail"]
    assert not api.uploads.exists()  # nothing written to the host's own disk


@pytest.mark.parametrize("bad_id", ["../../etc", "x" * 32, "0" * 31])
def test_upload_ids_cant_reach_anywhere_else(api, bad_id):
    assert api.get(f"/api/uploads/{bad_id}").status_code == 404
    assert api.delete(f"/api/uploads/{bad_id}", headers=HEADERS).status_code in (404, 405)


def test_another_website_cant_start_or_change_anything(api, wav_bytes):
    assert api.post("/api/uploads", json={"name": "a.wav", "size": 1000}).status_code == 403
    upload = start(api, wav_bytes)
    piece = pieces(wav_bytes)[0]
    assert api.put(f"/api/uploads/{upload['id']}/part", params={"offset": 0, "sha256": sha(piece)}, content=piece).status_code == 403
    assert api.delete(f"/api/uploads/{upload['id']}").status_code == 403
    assert api.post("/api/rescan").status_code == 403


# --- The watched folder -------------------------------------------------------------------


def test_a_file_being_copied_shows_as_arriving_until_it_settles(api, wav_bytes):
    folder = api.uploads.parent / "laptop"
    folder.mkdir()
    path = folder / "copying.wav"
    path.write_bytes(wav_bytes[:1000])
    library = api.app.state.library

    library.scan()
    body = api.get("/api/recordings").json()
    assert body["arriving"] == [{"name": "copying.wav", "size_bytes": 1000}]
    assert body["recordings"] == []

    path.write_bytes(wav_bytes[:2000])
    library.scan()
    assert api.get("/api/recordings").json()["arriving"][0]["size_bytes"] == 2000

    path.write_bytes(wav_bytes)
    library.scan()
    library.scan()
    body = api.get("/api/recordings").json()
    assert body["arriving"] == []
    assert [r["name"] for r in body["recordings"]] == ["copying.wav"]


def test_rescan_finds_finished_files_at_once_but_doesnt_rush_a_copy(api, wav_bytes):
    folder = api.uploads.parent / "laptop"
    folder.mkdir()
    done = folder / "copied_yesterday.wav"
    done.write_bytes(wav_bytes)
    old = time.time() - 3600
    os.utime(done, (old, old))
    growing = folder / "still_copying.wav"
    growing.write_bytes(wav_bytes[:1000])

    # Pressing it over and over is not the same as the size holding still between polls.
    for _ in range(3):
        assert api.post("/api/rescan", headers=HEADERS).status_code == 200
    body = api.get("/api/recordings").json()
    assert [r["name"] for r in body["recordings"]] == ["copied_yesterday.wav"]
    assert [a["name"] for a in body["arriving"]] == ["still_copying.wav"]


def test_part_files_are_never_listed(api, wav_bytes):
    api.uploads.mkdir()
    (api.uploads / "something.part").write_bytes(wav_bytes)
    (api.uploads / "something.wav.part").write_bytes(wav_bytes)
    library = api.app.state.library
    library.scan()
    library.scan()
    body = api.get("/api/recordings").json()
    assert body["recordings"] == [] and body["arriving"] == []


# --- A real server --------------------------------------------------------------------------

SERVER = textwrap.dedent(
    """
    import resource, sys, threading, time, uvicorn
    from iqscope.app import create_app
    radio, cache, mountinfo, port = sys.argv[1:5]
    app = create_app(radio, cache, mountinfo, static_dir=None, background=False)
    @app.get("/test/peak-kib")
    def peak() -> int:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    uvicorn.run(app, host="127.0.0.1", port=int(port), log_level="warning")
    """
)


@pytest.fixture
def server(tmp_path):
    radio = tmp_path / "radio"
    (radio / "recordings").mkdir(parents=True)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"50 22 0:40 / {radio} rw - cifs //host/share rw\n")
    (tmp_path / "cache").mkdir()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-c", SERVER, str(radio), str(tmp_path / "cache"), str(mountinfo), str(port)]
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"{base}/api/health", timeout=2)
                break
            except OSError:
                time.sleep(0.1)
        yield base, radio / "recordings" / "uploads"
    finally:
        process.terminate()
        process.wait(timeout=10)


def call(method: str, url: str, data: bytes | None = None, json_body: dict | None = None):
    headers = dict(HEADERS)
    if json_body is not None:
        data = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read() or b"null")


def test_a_large_upload_barely_changes_the_servers_memory(server):
    base, folder = server
    part = 32 * 2**20
    # 480 MiB, patterned so nothing can quietly be shared or compressed away.
    total = 15 * part
    frames = (total - 44) // 8
    header = (
        b"RIFF" + struct.pack("<I", 36 + frames * 8) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 2, 48_000, 48_000 * 8, 8, 32)
        + b"data" + struct.pack("<I", frames * 8)
    )
    total = len(header) + frames * 8
    upload = call("POST", f"{base}/api/uploads", json_body={"name": "big.wav", "size": total})
    assert upload["part_bytes"] == part

    baseline = call("GET", f"{base}/test/peak-kib")
    block = np.random.default_rng(1).bytes(part)
    whole = hashlib.sha256()
    hashes = []
    offset = 0
    while offset < total:
        length = min(part, total - offset)
        piece = (header + block)[:length] if offset == 0 else block[:length]
        hashes.append(sha(piece))
        whole.update(piece)
        call("PUT", f"{base}/api/uploads/{upload['id']}/part?offset={offset}&sha256={hashes[-1]}", data=piece)
        offset += length
    call("POST", f"{base}/api/uploads/{upload['id']}/finish", json_body={"pieces_sha256": piece_list_hash(hashes)})

    grew_mib = (call("GET", f"{base}/test/peak-kib") - baseline) / 1024
    # One piece in flight, not the file: anywhere near 480 MiB would be a failure.
    assert grew_mib < 100, f"memory grew {grew_mib:.0f} MiB"
    received = hashlib.sha256()
    with open(folder / "big.wav", "rb") as fh:
        while block := fh.read(2**20):
            received.update(block)
    assert received.hexdigest() == whole.hexdigest()


def test_a_connection_cut_mid_piece_leaves_the_upload_ready_to_continue(server):
    base, folder = server
    z, _ = synthesize("am", sample_rate=RATE, duration_s=1.0, offset_hz=30_000, snr_db=40.0)
    source = folder.parent / "source.wav.tmp"
    write_wav(source, z, RATE, sample_type="f32")
    data = source.read_bytes()
    source.unlink()
    upload = call("POST", f"{base}/api/uploads", json_body={"name": "cut.wav", "size": len(data)})
    host, port = base.removeprefix("http://").split(":")

    # Promise the whole piece, send half, and hang up.
    piece = data[: 32 * 2**20]
    with socket.create_connection((host, int(port))) as raw:
        raw.sendall(
            f"PUT /api/uploads/{upload['id']}/part?offset=0&sha256={sha(piece)} HTTP/1.1\r\n"
            f"Host: {host}\r\nX-IQScope: 1\r\nContent-Length: {len(piece)}\r\n\r\n".encode()
        )
        raw.sendall(piece[: len(piece) // 2])
        time.sleep(0.3)

    # The server lets go of the piece soon after, and no recording appears.
    deadline = time.time() + 10
    while True:
        try:
            result = call("PUT", f"{base}/api/uploads/{upload['id']}/part?offset=0&sha256={sha(piece)}", data=piece)
            break
        except urllib.error.HTTPError as err:
            assert err.code == 409 and time.time() < deadline
            time.sleep(0.2)
    assert result["received"] == len(piece)
    assert not list(folder.glob("*.wav"))
    call("POST", f"{base}/api/uploads/{upload['id']}/finish", json_body={"pieces_sha256": piece_list_hash([sha(piece)])})
    assert (folder / "cut.wav").read_bytes() == data
