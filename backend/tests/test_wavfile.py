"""Parser tests. Files are built byte by byte here, not with wavwrite, so the parser
is checked against the WAV format itself rather than against our own writer."""

import os
import struct

import numpy as np
import pytest

from iqscope.wavfile import MAX_READ_FRAMES, WavError, WavReader, read_header

GIB = 2**30


def chunk(chunk_id: bytes, body: bytes, size: int | None = None) -> bytes:
    pad = b"\0" if len(body) & 1 else b""
    return chunk_id + struct.pack("<I", len(body) if size is None else size) + body + pad


def fmt_chunk(tag=1, channels=2, rate=2_400_000, bits=16, align=None) -> bytes:
    align = channels * bits // 8 if align is None else align
    return chunk(b"fmt ", struct.pack("<HHIIHH", tag, channels, rate, rate * align, align, bits))


def riff(*chunks: bytes, magic=b"RIFF", size=None) -> bytes:
    body = b"WAVE" + b"".join(chunks)
    return magic + struct.pack("<I", len(body) if size is None else size) + body


def write(tmp_path, content: bytes, name="x.wav"):
    path = tmp_path / name
    path.write_bytes(content)
    return path


I16_PAIRS = struct.pack("<4h", 1000, -1000, 2000, -2000)
OK_PAIRS = np.array([[1000, -1000], [2000, -2000]], dtype=np.float32) / 32768


def read_all(path) -> np.ndarray:
    with WavReader(path) as reader:
        return reader.read(0, reader.header.frames)


# --- Sample types -------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag, bits, raw, expected",
    [
        (1, 8, bytes([0, 128, 255, 64]), [-1.0, 0.0, 127 / 128, -0.5]),
        (1, 16, struct.pack("<4h", -32768, 0, 16384, 32767), [-1.0, 0.0, 0.5, 32767 / 32768]),
        (1, 24, b"\x00\x00\x80\x00\x00\x00\x00\x00\x40\xff\xff\xff", [-1.0, 0.0, 0.5, -1 / 8388608]),
        (1, 32, struct.pack("<4i", -(2**31), 0, 2**30, -(2**29)), [-1.0, 0.0, 0.5, -0.25]),
        (3, 32, struct.pack("<4f", -1.0, 0.0, 0.5, 0.25), [-1.0, 0.0, 0.5, 0.25]),
    ],
)
def test_sample_types_read_back_expected_values(tmp_path, tag, bits, raw, expected):
    path = write(tmp_path, riff(fmt_chunk(tag=tag, bits=bits), chunk(b"data", raw)))
    got = read_all(path)
    assert got.dtype == np.float32
    assert got.shape == (2, 2)
    np.testing.assert_allclose(got.ravel(), expected, atol=1e-7)
    assert read_header(path).warnings == ()


def test_extensible_format_uses_subformat(tmp_path):
    align = 8
    body = struct.pack("<HHIIHH", 0xFFFE, 2, 2_400_000, 2_400_000 * align, align, 32)
    # cbSize, valid bits and channel mask come first, then the SubFormat GUID.
    extension = struct.pack("<HHI", 22, 32, 3) + struct.pack("<H", 3) + bytes(14)
    fmt = chunk(b"fmt ", body + extension)
    path = write(tmp_path, riff(fmt, chunk(b"data", struct.pack("<2f", 0.5, -0.5))))
    assert read_header(path).sample_type == "f32"
    np.testing.assert_allclose(read_all(path), [[0.5, -0.5]])


def test_unsupported_format_is_refused(tmp_path):
    path = write(tmp_path, riff(fmt_chunk(tag=0x55, bits=16), chunk(b"data", I16_PAIRS)))
    with pytest.raises(WavError, match="unsupported"):
        read_header(path)


def test_nan_and_infinity_become_finite(tmp_path):
    raw = struct.pack("<4f", float("nan"), float("inf"), float("-inf"), 0.5)
    path = write(tmp_path, riff(fmt_chunk(tag=3, bits=32), chunk(b"data", raw)))
    np.testing.assert_allclose(read_all(path).ravel(), [0.0, 1.0, -1.0, 0.5])


# --- Chunk walking ------------------------------------------------------------------


def test_metadata_before_and_after_data(tmp_path):
    path = write(
        tmp_path,
        riff(
            chunk(b"JUNK", bytes(28)),
            fmt_chunk(),
            chunk(b"LIST", b"odd"),  # odd size, so a pad byte follows it
            chunk(b"data", I16_PAIRS),
            chunk(b"LIST", b"tail metadata!"),
        ),
    )
    h = read_header(path)
    assert h.chunks == ("JUNK", "fmt", "LIST", "data", "LIST")
    assert h.data_offset != 44
    assert h.frames == 2
    assert h.warnings == ()
    np.testing.assert_allclose(read_all(path), OK_PAIRS)


def test_fmt_after_data_still_works(tmp_path):
    path = write(tmp_path, riff(chunk(b"data", I16_PAIRS), fmt_chunk()))
    np.testing.assert_allclose(read_all(path), OK_PAIRS)


def test_auxi_chunk_gives_center_frequency_and_time(tmp_path):
    start = struct.pack("<8H", 2021, 3, 0, 14, 12, 30, 45, 500)
    aux = start + bytes(16) + struct.pack("<I", 7_200_000) + bytes(4)
    path = write(tmp_path, riff(fmt_chunk(), chunk(b"auxi", aux), chunk(b"data", I16_PAIRS)))
    h = read_header(path)
    assert h.aux_center_hz == 7_200_000
    assert h.aux_started_at.isoformat() == "2021-03-14T12:30:45.500000"


def test_garbage_auxi_time_is_ignored(tmp_path):
    aux = bytes(32) + struct.pack("<I", 0) + bytes(4)
    path = write(tmp_path, riff(fmt_chunk(), chunk(b"auxi", aux), chunk(b"data", I16_PAIRS)))
    h = read_header(path)
    assert h.aux_center_hz is None
    assert h.aux_started_at is None


# --- RF64 and BW64 ------------------------------------------------------------------


def rf64_bytes(raw: bytes, magic=b"RF64", extra_after=b"") -> bytes:
    ds64 = chunk(b"ds64", struct.pack("<QQQI", 0, len(raw), len(raw) // 4, 0))
    return riff(ds64, fmt_chunk(), chunk(b"data", raw, size=0xFFFFFFFF), extra_after, magic=magic, size=0xFFFFFFFF)


@pytest.mark.parametrize("magic", [b"RF64", b"BW64"])
def test_rf64_small_file(tmp_path, magic):
    path = write(tmp_path, rf64_bytes(I16_PAIRS, magic=magic))
    h = read_header(path)
    assert h.container == magic.decode()
    assert h.frames == 2
    np.testing.assert_allclose(read_all(path), OK_PAIRS)


def test_rf64_ds64_size_stops_before_trailing_metadata(tmp_path):
    path = write(tmp_path, rf64_bytes(I16_PAIRS, extra_after=chunk(b"LIST", b"tail")))
    h = read_header(path)
    assert h.frames == 2
    assert h.chunks[-1] == "LIST"


def test_rf64_without_ds64_uses_file_size_and_warns(tmp_path):
    content = riff(fmt_chunk(), chunk(b"data", I16_PAIRS, size=0xFFFFFFFF), magic=b"RF64", size=0xFFFFFFFF)
    h = read_header(write(tmp_path, content))
    assert h.frames == 2
    assert any("ds64" in w for w in h.warnings)


def make_sparse(path, header: bytes, data_bytes: int):
    """Create a huge file without using disk space; unwritten regions read as zero."""
    path.write_bytes(header)
    os.truncate(path, len(header) + data_bytes)


def test_rf64_beyond_4gib_reads_correct_samples_at_the_far_end(tmp_path):
    data_bytes = 5 * GIB
    ds64 = chunk(b"ds64", struct.pack("<QQQI", 0, data_bytes, data_bytes // 4, 0))
    head = b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + ds64 + fmt_chunk()
    head += b"data" + struct.pack("<I", 0xFFFFFFFF)
    path = tmp_path / "long.wav"
    make_sparse(path, head, data_bytes)

    target = data_bytes // 4 - 2  # second-to-last pair, well past the 4 GiB line
    with open(path, "r+b") as fh:
        fh.seek(len(head) + target * 4)
        fh.write(struct.pack("<2h", 16384, -16384))

    with WavReader(path) as reader:
        assert reader.header.frames == data_bytes // 4
        assert reader.header.duration_s == pytest.approx(5 * GIB / 4 / 2_400_000)
        np.testing.assert_allclose(reader.read(target, 2), [[0.5, -0.5], [0, 0]])
        assert reader.read(target + 2, 10).shape == (0, 2)  # past the end


def test_ordinary_riff_that_wrapped_at_4gib_uses_real_length(tmp_path):
    data_bytes = int(4.5 * GIB)
    wrapped = data_bytes % 2**32
    head = b"RIFF" + struct.pack("<I", (wrapped + 36) % 2**32) + b"WAVE" + fmt_chunk()
    head += b"data" + struct.pack("<I", wrapped)
    path = tmp_path / "wrapped.wav"
    make_sparse(path, head, data_bytes)
    h = read_header(path)
    assert h.frames == data_bytes // 4
    assert any("wrapped" in w for w in h.warnings)


# --- Damaged headers ----------------------------------------------------------------


@pytest.mark.parametrize("bad_size", [0, 0xFFFFFFFF])
def test_crash_style_sizes_derive_length_from_file(tmp_path, bad_size):
    content = riff(fmt_chunk(), chunk(b"data", I16_PAIRS, size=bad_size), size=bad_size)
    path = write(tmp_path, content)
    h = read_header(path)
    assert h.frames == 2
    assert h.warnings
    np.testing.assert_allclose(read_all(path), OK_PAIRS)


def test_truncated_file_warns_and_reads_what_exists(tmp_path):
    content = riff(fmt_chunk(), chunk(b"data", I16_PAIRS, size=1_000_000))
    h = read_header(write(tmp_path, content))
    assert h.frames == 2
    assert any("truncated" in w for w in h.warnings)


def test_partial_final_frame_is_dropped(tmp_path):
    path = write(tmp_path, riff(fmt_chunk(), chunk(b"data", I16_PAIRS + b"\x01\x02\x03")))
    h = read_header(path)
    assert h.frames == 2
    assert any("whole frame" in w for w in h.warnings)
    assert read_all(path).shape == (2, 2)


def test_data_size_too_small_with_samples_after_it(tmp_path):
    samples = np.random.default_rng(0).integers(-20000, 20000, 400, dtype="<i2").tobytes()
    content = riff(fmt_chunk(), chunk(b"data", samples, size=8))
    h = read_header(write(tmp_path, content))
    assert h.frames == 200  # 400 int16 values make 200 pairs
    assert any("wrong or wrapped" in w for w in h.warnings)


def test_wrong_block_align_is_corrected(tmp_path):
    path = write(tmp_path, riff(fmt_chunk(align=7), chunk(b"data", I16_PAIRS)))
    h = read_header(path)
    assert h.frame_bytes == 4
    assert any("block size" in w for w in h.warnings)


def test_wrong_riff_size_warns_but_reads(tmp_path):
    path = write(tmp_path, riff(fmt_chunk(), chunk(b"data", I16_PAIRS), size=12345))
    h = read_header(path)
    assert h.frames == 2
    assert any("total size" in w for w in h.warnings)


def test_empty_data_chunk_is_valid(tmp_path):
    h = read_header(write(tmp_path, riff(fmt_chunk(), chunk(b"data", b""))))
    assert h.frames == 0
    assert h.warnings == ()


@pytest.mark.parametrize(
    "content, message",
    [
        (b"", "not a RIFF"),
        (b"RIFF", "not a RIFF"),
        (b"OggS" + bytes(40), "not a RIFF"),
        (b"RIFF\0\0\0\0AVI " + bytes(40), "not a RIFF"),
        (riff(chunk(b"data", I16_PAIRS)), "no fmt"),
        (riff(fmt_chunk()), "no data"),
        (riff(chunk(b"fmt ", b"short"), chunk(b"data", I16_PAIRS)), "too short"),
        (riff(fmt_chunk(channels=0), chunk(b"data", I16_PAIRS)), "invalid format"),
        (riff(fmt_chunk(rate=0), chunk(b"data", I16_PAIRS)), "invalid format"),
    ],
)
def test_unusable_files_are_refused_clearly(tmp_path, content, message):
    with pytest.raises(WavError, match=message):
        read_header(write(tmp_path, content))


def test_metadata_chunk_with_huge_size_does_not_hang_or_crash(tmp_path):
    content = riff(fmt_chunk(), chunk(b"LIST", b"abc", size=0xFFFFFFF0), chunk(b"data", I16_PAIRS))
    with pytest.raises(WavError, match="no data"):
        read_header(write(tmp_path, content))


# --- Random access ------------------------------------------------------------------


@pytest.fixture
def counting_file(tmp_path):
    """1000 pairs where pair n is (n, -n), so any position can be checked."""
    raw = np.column_stack((np.arange(1000), -np.arange(1000))).astype("<i2").tobytes()
    return write(tmp_path, riff(fmt_chunk(), chunk(b"data", raw)))


def test_read_from_the_middle(counting_file):
    with WavReader(counting_file) as reader:
        got = reader.read(500, 3) * 32768
    np.testing.assert_allclose(got, [[500, -500], [501, -501], [502, -502]])


def test_read_past_the_end_returns_fewer(counting_file):
    with WavReader(counting_file) as reader:
        assert reader.read(998, 10).shape == (2, 2)
        assert reader.read(1000, 10).shape == (0, 2)
        assert reader.read(5000, 10).shape == (0, 2)


def test_invalid_reads_are_refused(counting_file):
    with WavReader(counting_file) as reader:
        with pytest.raises(ValueError):
            reader.read(-1, 10)
        with pytest.raises(ValueError):
            reader.read(0, -1)
        with pytest.raises(ValueError, match="limit"):
            reader.read(0, MAX_READ_FRAMES + 1)


def test_file_shrinking_after_open_returns_what_remains(counting_file):
    with WavReader(counting_file) as reader:
        os.truncate(counting_file, os.path.getsize(counting_file) - 400)
        assert reader.read(0, 1000).shape == (900, 2)
