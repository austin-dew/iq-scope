#!/usr/bin/env python3
"""Measure whether a storage path can keep up with raw IQ playback.

It reads the way the app will: 1 second of IQ at a time (9.6 MB at 2.4 MS/s Int16).
  sequential  one chunk after another, like normal playback
  random      chunks at scattered positions, like scrubbing

O_DIRECT skips this machine's file cache, so numbers reflect the path to the disk. For
the NAS's cache to be cold as well, clear it first.
Standard library only, and read-only, so it runs directly on a host.

Usage: read_speed.py FILE [--seconds N]
"""

from __future__ import annotations

import argparse
import mmap
import os
import random
import statistics
import time

CHUNK_BYTES = 9_600_000
REQUIRED_MB_S = CHUNK_BYTES / 1e6
ALIGN = 4096


def open_reader(path: str) -> tuple[int, bool]:
    """Open with O_DIRECT if the filesystem allows it, and say whether it did."""
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECT), True
    except OSError:
        return os.open(path, os.O_RDONLY), False


def read_chunk(fd: int, offset: int, size: int, buffer: mmap.mmap, direct: bool) -> int:
    """Read one chunk. O_DIRECT needs aligned offsets and buffers; mmap memory is aligned."""
    if not direct:
        return len(os.pread(fd, size, offset))
    done = 0
    while done < size:
        got = os.preadv(fd, [memoryview(buffer)[: min(len(buffer), size - done)]], offset + done)
        if got <= 0:
            break
        done += got
    return done


def report(name: str, times: list[float], total_bytes: int) -> None:
    elapsed = sum(times)
    rate = total_bytes / elapsed / 1e6
    ms = sorted(t * 1000 for t in times)
    print(
        f"{name:<11} {rate:7.1f} MB/s  ({rate / REQUIRED_MB_S:4.1f}x real time)   "
        f"per chunk: median {statistics.median(ms):6.1f} ms, slowest {ms[-1]:7.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--seconds", type=int, default=60, help="seconds of IQ to read in each test")
    args = parser.parse_args()

    size = os.path.getsize(args.file)
    chunk = CHUNK_BYTES // ALIGN * ALIGN
    last = (size - chunk) // ALIGN
    if last <= 0:
        raise SystemExit("file is too small to test")

    fd, direct = open_reader(args.file)
    buffer = mmap.mmap(-1, 1 << 22)  # 4 MiB per request, the usual SMB maximum
    print(f"{args.file}: {size / 2**30:.2f} GiB, O_DIRECT {'on' if direct else 'UNAVAILABLE (results may be cached)'}")
    print(f"Playback needs {REQUIRED_MB_S} MB/s at 2.4 MS/s Int16.")

    count = min(args.seconds, size // chunk)
    start_chunk = random.randrange(0, max(size // chunk - count, 1))
    times = []
    for i in range(count):
        started = time.perf_counter()
        read_chunk(fd, (start_chunk + i) * chunk, chunk, buffer, direct)
        times.append(time.perf_counter() - started)
    report("sequential", times, count * chunk)

    times = []
    for _ in range(min(args.seconds, 30)):
        offset = random.randrange(0, last) * ALIGN
        started = time.perf_counter()
        read_chunk(fd, offset, chunk, buffer, direct)
        times.append(time.perf_counter() - started)
    report("random", times, len(times) * chunk)
    os.close(fd)


if __name__ == "__main__":
    main()
