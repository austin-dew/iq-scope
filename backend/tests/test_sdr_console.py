"""SDR Console's metadata, plus the user's long recordings from the RECORDINGS SD card.

The card tests skip when it isn't mounted. Expected values come from reading the
headers by hand: a hex dump and the XML text stored in each file.
"""

import struct
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import pytest

from iqscope.recording import frequency_from_name, open_recording
from iqscope.wavfile import read_header

# A folder of real SDR Console captures, if you have one. These tests skip without it.
CARD = Path(os.environ.get("IQSCOPE_TEST_CARD", "/nonexistent"))
IO_B = CARD / "Jupiter Experiments/Io B"
XML = (
    '<?xml version="1.0"?><SDR-XML-Root Description="Saved recording data">'
    '<Definition RadioModel="RTL-SDR Blog V4" SoftwareName="SDR Console" '
    'RadioCenterFreq="22002000" SampleRate="2400000" UTCSeconds="1772075075"/></SDR-XML-Root>'
)


def build(auxi: bytes) -> bytes:
    def chunk(chunk_id, body):
        return chunk_id + struct.pack("<I", len(body)) + body + (b"\0" if len(body) & 1 else b"")

    fmt = struct.pack("<HHIIHH", 1, 2, 2_400_000, 9_600_000, 4, 16)
    body = b"WAVE" + chunk(b"fmt ", fmt) + chunk(b"auxi", auxi) + chunk(b"data", bytes(8))
    return b"RIFF" + struct.pack("<I", len(body)) + body


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-8"])
def test_xml_auxi_gives_frequency_time_and_radio(tmp_path, encoding):
    path = tmp_path / "x.wav"
    path.write_bytes(build(XML.encode(encoding)))
    h = read_header(path)
    assert h.aux_center_hz == 22_002_000
    assert h.aux_started_at == datetime(2026, 2, 26, 3, 4, 35, tzinfo=timezone.utc)
    assert h.aux_recorded_with == "RTL-SDR Blog V4, SDR Console"


def test_xml_is_never_misread_as_binary_fields(tmp_path):
    """The bug this guards against: XML text at offset 32 reads as about 3.1 MHz."""
    path = tmp_path / "x.wav"
    no_frequency = '<?xml version="1.0"?><SDR-XML-Root Description="nothing useful here at all"/>'
    path.write_bytes(build(no_frequency.encode("utf-16-le")))
    assert read_header(path).aux_center_hz is None


@pytest.mark.parametrize(
    "attribute",
    ['RadioCenterFreq="abc"', 'RadioCenterFreq="-5"', 'RadioCenterFreq=""', 'RadioCenterFreq="0"'],
)
def test_bad_xml_frequency_is_ignored(tmp_path, attribute):
    path = tmp_path / "x.wav"
    path.write_bytes(build(f"<SDR-XML-Root {attribute}/>".encode("utf-8")))
    assert read_header(path).aux_center_hz is None


def test_absurd_xml_time_is_ignored(tmp_path):
    path = tmp_path / "x.wav"
    path.write_bytes(build(('<SDR-XML-Root UTCSeconds="' + "9" * 40 + '"/>').encode("utf-8")))
    assert read_header(path).aux_started_at is None


def test_binary_auxi_without_a_valid_time_is_not_trusted(tmp_path):
    path = tmp_path / "x.wav"
    path.write_bytes(build(b"\x7f" * 32 + struct.pack("<I", 7_200_000) + bytes(8)))
    assert read_header(path).aux_center_hz is None


def test_auxi_frequency_beats_the_filename(tmp_path):
    path = tmp_path / "renamed_10mhz.wav"
    path.write_bytes(build(XML.encode("utf-16-le")))
    with open_recording(path) as rec:
        assert rec.frequency_hz == 22_002_000
        assert rec.frequency_source == "auxi chunk"


def test_filename_with_several_frequencies_is_unknown_and_explained(tmp_path):
    name = "Ex 8_10mhz_20.1mhz_28.2mhz_Day.wav"
    assert frequency_from_name(name) is None
    path = tmp_path / name
    from iqscope.wavwrite import write_wav

    write_wav(path, np.zeros((16, 2)), 2_400_000)
    with open_recording(path) as rec:
        assert rec.frequency_hz is None
        assert "20,100,000 Hz" in rec.frequency_note


def test_repeated_frequency_in_a_filename_is_not_a_conflict():
    assert frequency_from_name("22mhz_copy_of_22MHz.wav") == 22_000_000


# --- The user's real files ----------------------------------------------------------


def need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path} is not available")
    return path


@pytest.mark.parametrize(
    "name, frames, data_offset_end",
    [
        ("Jupiter_Storm_Watch_02-26-2026_Io_B.wav", 30_240_192_000, 120_960_770_014),
        ("Jupiter_Storm_Watch_02-27-2026_Io_B_After.wav", 30_260_412_000, 121_041_650_014),
    ],
)
def test_real_112_gib_sdr_console_rf64(name, frames, data_offset_end):
    with open_recording(need(IO_B / name)) as rec:
        h = rec.header
        assert h.container == "RF64"
        assert h.chunks == ("ds64", "fmt", "auxi", "data", "LIST")
        assert h.frames == frames
        assert h.data_offset + h.data_bytes == data_offset_end
        assert h.warnings == ()
        assert rec.kind == "iq"
        assert rec.frequency_hz == 22_002_000
        assert rec.frequency_source == "auxi chunk"
        assert h.aux_recorded_with == "RTL-SDR Blog V4, SDR Console"
        # About 3.5 hours in. It must return real signal, not zeros from a bad offset.
        tail = rec.read_iq(h.frames - 200_000, 100_000)
        assert tail.size == 100_000
        assert np.abs(tail).max() > 0


def test_real_rf64_start_time_matches_its_stored_utc_text():
    with open_recording(need(IO_B / "Jupiter_Storm_Watch_02-26-2026_Io_B.wav")) as rec:
        assert rec.started_at == datetime(2026, 2, 26, 3, 4, 35, tzinfo=timezone.utc)


def test_real_file_whose_size_wrapped_at_4_gib():
    """Its header claims 1,468,984,320 data bytes; adding 2**32 gives the true size."""
    path = need(CARD / "Jupiter Experiments/Jupiter Ex 1_22mhz_Night_4_10min.wav")
    with open_recording(path) as rec:
        h = rec.header
        assert h.container == "RIFF"
        assert h.data_bytes == 1_468_984_320 + 2**32
        assert h.frames == 1_440_987_904
        assert h.duration_s == pytest.approx(600.41, abs=0.01)  # the "10 min" in its name
        assert any("wrapped" in w for w in h.warnings)
        assert rec.frequency_hz == 22_000_000
