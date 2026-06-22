"""Playback controls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ViewMode(Enum):
    PLAYING = "playing"
    PAUSED = "paused"
    BUFFER = "buffer"


@dataclass
class PlaybackState:
    mode: ViewMode = ViewMode.PLAYING
    frame_pos: int = 0
    buffer_index: int = 0

    def is_analyzing(self) -> bool:
        return self.mode != ViewMode.PLAYING

    def is_frozen(self) -> bool:
        return self.mode != ViewMode.PLAYING


ON_SCREEN_CONTROLS: list[tuple[str, str]] = [
    ("Space", "Play / Pause"),
    ("A / D", "Step 5 frames"),
    ("J / K", "Step 1 frame"),
    ("Home", "Go to start"),
    ("End", "Go to end"),
    ("R", "Restart video"),
    ("B", "Scrub buffer"),
    ("Esc", "Exit buffer"),
    ("S", "Save clip"),
    ("Q", "Quit"),
]


def decode_key(key_raw: int) -> str | int:
    if key_raw == -1:
        return ""
    special = {
        2424832: "left",
        2555904: "right",
        2359296: "home",
        2293760: "end",
    }
    if key_raw in special:
        return special[key_raw]
    key = key_raw & 0xFF
    if key == 27:
        return "esc"
    return chr(key) if key else ""


def handle_playback_key(
    key_input: str | int,
    playback: PlaybackState,
    *,
    is_live: bool,
    buffer_len: int,
    total_frames: int,
) -> str | None:
    if not key_input:
        return None

    if key_input in ("q", "Q"):
        return "quit"

    if key_input == " ":
        if playback.mode == ViewMode.PLAYING:
            playback.mode = ViewMode.PAUSED
        elif playback.mode == ViewMode.PAUSED:
            playback.mode = ViewMode.PLAYING
        elif playback.mode == ViewMode.BUFFER:
            playback.mode = ViewMode.PAUSED
        return "pause"

    if key_input in ("r", "R") and not is_live:
        playback.frame_pos = 0
        playback.mode = ViewMode.PAUSED
        return "restart"

    if key_input in ("b", "B") and buffer_len > 0:
        playback.mode = ViewMode.BUFFER
        playback.buffer_index = buffer_len - 1
        return "buffer"

    if key_input == "esc" and playback.mode == ViewMode.BUFFER:
        playback.mode = ViewMode.PAUSED
        return "exit_buffer"

    step, single = 5, 1

    if key_input in ("a", "left", "j"):
        delta = single if key_input == "j" else step
        if playback.mode == ViewMode.BUFFER and buffer_len:
            playback.buffer_index = max(0, playback.buffer_index - delta)
        elif not is_live:
            playback.frame_pos = max(0, playback.frame_pos - delta)
            playback.mode = ViewMode.PAUSED
        return "step_back"

    if key_input in ("d", "right", "k"):
        delta = single if key_input == "k" else step
        if playback.mode == ViewMode.BUFFER and buffer_len:
            playback.buffer_index = min(buffer_len - 1, playback.buffer_index + delta)
        elif not is_live:
            playback.frame_pos = min(max(total_frames - 1, 0), playback.frame_pos + delta)
            playback.mode = ViewMode.PAUSED
        return "step_forward"

    if key_input == "home":
        if playback.mode == ViewMode.BUFFER:
            playback.buffer_index = 0
        elif not is_live:
            playback.frame_pos = 0
            playback.mode = ViewMode.PAUSED
        return "jump_start"

    if key_input == "end":
        if playback.mode == ViewMode.BUFFER and buffer_len:
            playback.buffer_index = buffer_len - 1
        elif not is_live:
            playback.frame_pos = max(total_frames - 1, 0)
            playback.mode = ViewMode.PAUSED
        return "jump_end"

    if key_input in ("s", "S"):
        return "save_clip"

    return None
