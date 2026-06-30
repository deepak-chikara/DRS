#!/usr/bin/env python3
"""Download test videos from URLs in tools/test_video_urls.txt (requires yt-dlp)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL_FILE = Path(__file__).resolve().parent / "test_video_urls.txt"
OUT_DIR = ROOT / "tests" / "fixtures" / "videos"


def _read_urls() -> list[str]:
    if not URL_FILE.is_file():
        return []
    urls = []
    for line in URL_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def main() -> int:
    urls = _read_urls()
    if not urls:
        print(f"No URLs in {URL_FILE}")
        print("Add YouTube links (one per line), then run again.")
        print("See docs/TEST_VIDEOS.md for search suggestions.")
        return 1

    if shutil.which("yt-dlp") is None:
        print("yt-dlp not found. Install with: pip install yt-dlp")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(urls)} clip(s) to {OUT_DIR}")

    for i, url in enumerate(urls, start=1):
        out_template = str(OUT_DIR / f"lbw_test_{i:02d}.%(ext)s")
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", out_template,
            "--no-playlist",
            "--max-filesize", "80M",
            url,
        ]
        print(f"\n[{i}/{len(urls)}] {url}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Failed: {exc}", file=sys.stderr)
            continue

    print(f"\nDone. Open in DRS Pro: File → Open Video → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
