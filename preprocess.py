"""
Validate and summarize face-crop images under dataset/real and dataset/fake.
Removes nothing by default; use --remove-bad to delete unreadable files (optional backup).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from src.config import DATA_DIR


def scan(root: Path, remove_bad: bool) -> tuple[int, int, list[str]]:
    ok, bad = 0, 0
    bad_paths: list[str] = []
    for sub in ("real", "fake"):
        d = root / sub
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    if w < 32 or h < 32:
                        raise ValueError("too small")
                ok += 1
            except Exception as e:
                bad += 1
                bad_paths.append(f"{p}: {e}")
                if remove_bad:
                    p.unlink(missing_ok=True)
    return ok, bad, bad_paths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    p.add_argument(
        "--remove-bad",
        action="store_true",
        help="Delete files that fail to load (use with care)",
    )
    args = p.parse_args(argv)
    root = Path(args.data_dir)
    ok, bad, bad_paths = scan(root, args.remove_bad)
    print(f"OK: {ok}  Bad: {bad}  root={root.resolve()}")
    if bad_paths:
        print("--- failures (first 20) ---")
        for line in bad_paths[:20]:
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
