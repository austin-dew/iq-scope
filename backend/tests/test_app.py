"""The health check must tell the truth when the share is down or stuck."""

import threading

import pytest
from fastapi.testclient import TestClient

from iqscope import app as app_module
from iqscope.app import ShareProbe, create_app, filesystem_type


def mountinfo_file(tmp_path, radio_dir, fstype):
    path = tmp_path / "mountinfo"
    path.write_text(
        "22 1 8:1 / / rw,noatime shared:1 - ext4 /dev/root rw\n"
        f"50 22 0:40 /mnt/radio {radio_dir} rw,noatime master:9 - {fstype} //host/share/radio rw\n"
    )
    return str(path)


@pytest.fixture
def radio(tmp_path):
    (tmp_path / "radio/recordings").mkdir(parents=True)
    return tmp_path / "radio"


def client(tmp_path, radio, fstype):
    info = mountinfo_file(tmp_path, str(radio), fstype)
    return TestClient(create_app(str(radio), str(tmp_path), info, static_dir=None, background=False))


def test_filesystem_type_reads_the_last_mount_at_that_point(tmp_path):
    path = tmp_path / "mountinfo"
    path.write_text(
        "50 22 0:40 / /radio rw - ext4 /dev/root rw\n"
        "51 50 0:41 / /radio rw master:9 - cifs //host/share rw\n"
        "52 22 0:42 / /radio/other rw - xfs /dev/x rw\n"
    )
    assert filesystem_type("/radio", str(path)) == "cifs"
    assert filesystem_type("/nothing", str(path)) is None


def test_share_online(tmp_path, radio):
    body = client(tmp_path, radio, "cifs").get("/api/health").json()
    assert body["status"] == "ok"
    assert body["share"]["readable"] is True
    assert body["cache_writable"] is True


def test_share_down_is_reported_even_though_the_bind_mount_exists(tmp_path, radio):
    """With the share unmounted, the container still sees a mount: the host's empty folder."""
    res = client(tmp_path, radio, "ext4").get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "degraded"
    assert body["share"]["mounted"] is False
    assert "ext4" in body["share"]["detail"]


def test_mounted_share_without_recordings_folder_is_not_healthy(tmp_path):
    empty = tmp_path / "radio"
    empty.mkdir()
    body = client(tmp_path, empty, "cifs").get("/api/health").json()
    assert body["share"]["mounted"] is True
    assert body["share"]["readable"] is False


def test_read_error_is_reported_not_raised(tmp_path, radio, monkeypatch):
    def fail(_path):
        raise OSError(112, "Host is down")

    monkeypatch.setattr(app_module.os, "listdir", fail)
    body = client(tmp_path, radio, "cifs").get("/api/health").json()
    assert body["status"] == "degraded"
    assert "Host is down" in body["share"]["detail"]


def test_stuck_share_times_out_and_later_checks_do_not_pile_up(tmp_path, radio, monkeypatch):
    release = threading.Event()
    calls = []

    def stuck(_path):
        calls.append(1)
        release.wait(30)
        return []

    monkeypatch.setattr(app_module.os, "listdir", stuck)
    monkeypatch.setattr(app_module, "_PROBE_TIMEOUT_S", 0.05)
    probe = ShareProbe(str(radio), mountinfo_file(tmp_path, str(radio), "cifs"))
    try:
        for _ in range(3):
            result = probe.check()
            assert result["readable"] is False
            assert "didn't answer" in result["detail"]
        assert len(calls) == 1  # only one read was ever started
    finally:
        release.set()
    # Once the share answers again, the next checks recover without a restart.
    probe._pending.result(timeout=5)
    monkeypatch.setattr(app_module.os, "listdir", lambda _p: [])
    assert probe.check()["readable"] is True


def test_api_docs_are_not_exposed(tmp_path, radio):
    c = client(tmp_path, radio, "cifs")
    assert c.get("/docs").status_code == 404
    assert c.get("/openapi.json").status_code == 404
