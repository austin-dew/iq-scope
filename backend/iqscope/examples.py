"""Built-in example clips: the same melody through each mode, for comparing them.

They're generated once, on the server's own disk, so they never touch the recordings.
Each file is written under a temporary name and renamed, so a half-made one is never seen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .synth import MODES, synthesize, synthesize_mode_switch
from .wavwrite import write_wav

CENTER_HZ = 7_200_000
SAMPLE_RATE = 1_024_000
OFFSET_HZ = 50_000
CLIP_SECONDS = 5.0
SEGMENT_SECONDS = 2.0


@dataclass(frozen=True)
class Example:
    file: str
    title: str
    mode: str
    description: str


_DESCRIPTIONS = {
    "am": ("AM", "The arrow's length follows the melody, and a steady carrier sits in the middle."),
    "usb": ("USB", "Upper sideband only, with no carrier. Try it as LSB and it disappears."),
    "lsb": ("LSB", "Lower sideband only, with no carrier. Try it as USB and it disappears."),
    "cw": ("CW", "Morse code for IQ. The arrow switches on and off."),
    "nfm": ("NFM", "The arrow's spin speed follows the melody, and its length holds steady."),
    "wfm": ("WFM", "The same idea as NFM with a far wider swing, so five pairs make each audio sample."),
}

EXAMPLES = [
    Example(f"example_{mode}_{CENTER_HZ}Hz.wav", _DESCRIPTIONS[mode][0], mode, _DESCRIPTIONS[mode][1]) for mode in MODES
]
SWITCH = Example(
    f"example_switch_{CENTER_HZ}Hz.wav",
    "ALL MODES IN TURN",
    "am",
    f"The same tone through every mode, {SEGMENT_SECONDS:g} seconds each. The mode follows along as it plays.",
)
_BY_FILE = {e.file: e for e in [*EXAMPLES, SWITCH]}


def example_info(file_name: str) -> dict | None:
    """What the app needs to open this example, tuned and ready to compare."""
    example = _BY_FILE.get(file_name)
    if example is None:
        return None
    info = {
        "title": example.title,
        "mode": example.mode,
        "description": example.description,
        "offset_hz": OFFSET_HZ,
        "order": [*EXAMPLES, SWITCH].index(example),
    }
    if example is SWITCH:
        info["segments"] = [
            {"mode": mode, "start_s": i * SEGMENT_SECONDS, "end_s": (i + 1) * SEGMENT_SECONDS}
            for i, mode in enumerate(MODES)
        ]
    return info


def _write(folder: Path, name: str, z) -> None:
    temp = folder / f".{name}.{os.getpid()}.tmp"
    write_wav(temp, z, SAMPLE_RATE)
    temp.replace(folder / name)


def ensure_examples(folder: str | os.PathLike) -> list[str]:
    """Generate whichever clips are missing. Returns the names it made."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    made = []
    for example in EXAMPLES:
        if not (folder / example.file).exists():
            z, _ = synthesize(
                example.mode, sample_rate=SAMPLE_RATE, duration_s=CLIP_SECONDS, center_hz=CENTER_HZ,
                offset_hz=OFFSET_HZ, message="melody",
            )
            _write(folder, example.file, z)
            made.append(example.file)
    if not (folder / SWITCH.file).exists():
        z, _ = synthesize_mode_switch(
            sample_rate=SAMPLE_RATE, segment_s=SEGMENT_SECONDS, center_hz=CENTER_HZ, offset_hz=OFFSET_HZ,
        )
        _write(folder, SWITCH.file, z)
        made.append(SWITCH.file)
    return made
