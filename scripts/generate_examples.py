#!/usr/bin/env python3
"""Generate the labeled example clips into examples/ (not stored in git).

The app makes its own copy on the server; this is for listening or testing locally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iqscope.examples import ensure_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "examples")
    args = parser.parse_args()
    made = ensure_examples(args.out)
    print(f"made {len(made)} clips in {args.out}" if made else f"nothing to do in {args.out}")


if __name__ == "__main__":
    main()
