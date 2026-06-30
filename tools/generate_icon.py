"""Generate DRS Pro icon — top-down LBW corridor (pitch + yellow lines + red ball)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    chunk = tag + data
    return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)


def _in_rect(x: float, y: float, x0: float, y0: float, x1: float, y1: float) -> bool:
    return x0 <= x <= x1 and y0 <= y <= y1


def _in_round_rect(x: float, y: float, x0: float, y0: float, x1: float, y1: float, r: float) -> bool:
    if not _in_rect(x, y, x0 + r, y0, x1 - r, y1):
        if not _in_rect(x, y, x0, y0 + r, x1, y1 - r):
            corners = (
                (x0 + r, y0 + r, r),
                (x1 - r, y0 + r, r),
                (x0 + r, y1 - r, r),
                (x1 - r, y1 - r, r),
            )
            return any((x - cx) ** 2 + (y - cy) ** 2 <= rad ** 2 for cx, cy, rad in corners)
    return True


def _on_line(x: float, y: float, lx: float, y0: float, y1: float, half_w: float) -> bool:
    return abs(x - lx) <= half_w and y0 <= y <= y1


def _render_icon_rgba(size: int) -> bytes:
    """
    Top-down LBW corridor:
    - White circular badge (visible on dark title bars)
    - Green pitch strip (bowler → striker)
    - Two yellow parallel lines = stump corridor
    - Red ball on the wicket line
    """
    pixels = bytearray(size * size * 4)
    cx, cy = size / 2, size / 2
    outer_r = size * 0.47
    inner_r = size * 0.44

    pitch_w = size * 0.28
    pitch_h = size * 0.62
    px0 = cx - pitch_w / 2
    px1 = cx + pitch_w / 2
    py0 = cy - pitch_h / 2 + size * 0.02
    py1 = cy + pitch_h / 2 - size * 0.02
    corner_r = max(1.0, size * 0.04)

    corridor_gap = pitch_w * 0.38
    off_x = cx - corridor_gap / 2
    leg_x = cx + corridor_gap / 2
    line_w = max(1.2, size * 0.035)

    ball_x = cx
    ball_y = cy + pitch_h * 0.12
    ball_r = max(1.5, size * 0.075)

    for y in range(size):
        for x in range(size):
            i = (y * size + x) * 4
            fx, fy = float(x), float(y)
            dist = ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5
            r, g, b, a = 0, 0, 0, 0

            if dist <= outer_r:
                r, g, b, a = 255, 255, 255, 255
            if dist <= inner_r:
                r, g, b, a = 245, 247, 250, 255

            if _in_round_rect(fx, fy, px0, py0, px1, py1, corner_r):
                r, g, b, a = 56, 142, 60, 255

            # Crease lines at each end
            crease_h = max(1.0, size * 0.018)
            if _in_rect(fx, fy, px0, py0, px1, py0 + crease_h):
                r, g, b = 250, 250, 250
            if _in_rect(fx, fy, px0, py1 - crease_h, px1, py1):
                r, g, b = 250, 250, 250

            # Stump corridor (yellow) — full length of pitch
            if _on_line(fx, fy, off_x, py0 + crease_h, py1 - crease_h, line_w):
                r, g, b = 255, 213, 79
            if _on_line(fx, fy, leg_x, py0 + crease_h, py1 - crease_h, line_w):
                r, g, b = 255, 213, 79

            # Wicket line hint (faint white center)
            if _on_line(fx, fy, cx, py0, py1, max(0.6, size * 0.008)):
                r, g, b = 200, 230, 200

            # Red ball on wicket line
            if (fx - ball_x) ** 2 + (fy - ball_y) ** 2 <= ball_r ** 2:
                r, g, b = 229, 57, 53
            if (fx - ball_x + ball_r * 0.3) ** 2 + (fy - ball_y - ball_r * 0.3) ** 2 <= (ball_r * 0.25) ** 2:
                r, g, b = 255, 255, 255

            pixels[i : i + 4] = bytes((b, g, r, a))

    return bytes(pixels)


def _write_png(path: Path, size: int = 256) -> None:
    rows = []
    raw = _render_icon_rgba(size)
    for y in range(size):
        row = bytearray([0])
        off = y * size * 4
        for x in range(size):
            row.extend(raw[off + x * 4 : off + x * 4 + 3])
        rows.append(bytes(row))
    compressed = zlib.compress(b"".join(rows), 9)
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", compressed)
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _bmp_header(size: int, data_size: int) -> bytes:
    header_size = 40
    file_size = 14 + header_size + data_size
    return struct.pack(
        "<2sIHHIiiHHIIiiII",
        b"BM",
        file_size,
        0,
        14 + header_size,
        header_size,
        size,
        size * 2,
        1,
        32,
        0,
        data_size,
        0,
        0,
        0,
        0,
    )


def _icon_image_bmp(size: int) -> bytes:
    rgba = _render_icon_rgba(size)
    row_bytes = size * 4
    pad = (4 - (row_bytes % 4)) % 4
    pixel_data = bytearray()
    for y in range(size - 1, -1, -1):
        row = bytearray()
        for x in range(size):
            off = (y * size + x) * 4
            row.extend(rgba[off : off + 3])
            row.append(rgba[off + 3])
        row.extend(b"\x00" * pad)
        pixel_data.extend(row)
    and_mask = b"\x00" * (((size + 31) // 32) * 4 * size)
    return _bmp_header(size, len(pixel_data) + len(and_mask)) + bytes(pixel_data) + and_mask


def _write_multi_ico(path: Path, sizes: tuple[int, ...] = (16, 32, 48, 64, 128, 256)) -> None:
    images = [(s, _icon_image_bmp(s)) for s in sizes]
    offset = 6 + 16 * len(images)
    header = struct.pack("<HHH", 0, 1, len(images))
    entries = bytearray()
    for size, data in images:
        w = 0 if size >= 256 else size
        h = 0 if size >= 256 else size
        entries.extend(struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, len(data), offset))
        offset += len(data)
    out = header + bytes(entries)
    for _, data in images:
        out += data
    path.write_bytes(out)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    png = assets / "icon.png"
    ico = assets / "icon.ico"
    _write_png(png, 256)
    _write_multi_ico(ico)
    print(f"Wrote LBW corridor icon: {ico} and {png}")


if __name__ == "__main__":
    main()
