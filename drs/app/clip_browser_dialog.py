"""Browse saved DRS clips and match recordings."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from drs.paths import user_clips_dir, user_data_dir, user_matches_dir


def _fmt_time(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return ""


class ClipBrowserDialog(QDialog):
    def __init__(self, parent=None, *, on_open_video=None):
        super().__init__(parent)
        self._on_open_video = on_open_video
        self.setWindowTitle("Clips & Recordings")
        self.setMinimumSize(560, 420)

        tabs = QTabWidget()
        tabs.addTab(self._build_clips_tab(), "DRS Clips")
        tabs.addTab(self._build_matches_tab(), "Match Recordings")
        tabs.addTab(self._build_calls_tab(), "DRS Call JSON")

        open_btn = QPushButton("Open selected video")
        open_btn.clicked.connect(self._open_selected)
        folder_btn = QPushButton("Open folder in Explorer")
        folder_btn.clicked.connect(self._open_folder)

        row = QHBoxLayout()
        row.addWidget(open_btn)
        row.addWidget(folder_btn)
        row.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(f"Saved under: {user_data_dir()}")
        )
        layout.addWidget(tabs)
        layout.addLayout(row)
        layout.addWidget(buttons)
        self._tabs = tabs

    def _build_clips_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self._clips_list = QListWidget()
        for path in sorted(user_clips_dir().glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
            item = QListWidgetItem(f"{path.name}  ({_fmt_time(path)})")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self._clips_list.addItem(item)
        if self._clips_list.count() == 0:
            self._clips_list.addItem("(No clips yet — press F9 during live match)")
        layout.addWidget(self._clips_list)
        return w

    def _build_matches_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self._matches_list = QListWidget()
        matches = user_matches_dir()
        for path in sorted(matches.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
            rel = path.relative_to(matches)
            item = QListWidgetItem(f"{rel}  ({_fmt_time(path)})")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self._matches_list.addItem(item)
        if self._matches_list.count() == 0:
            self._matches_list.addItem("(No match recordings yet)")
        layout.addWidget(self._matches_list)
        return w

    def _build_calls_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self._calls_list = QListWidget()
        for path in sorted(user_clips_dir().glob("*_call.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            summary = path.name
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                v = data.get("verdict", data.get("cv_verdict", ""))
                summary = f"{path.name} — {v}"
            except (json.JSONDecodeError, OSError):
                pass
            item = QListWidgetItem(f"{summary}  ({_fmt_time(path)})")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self._calls_list.addItem(item)
        if self._calls_list.count() == 0:
            self._calls_list.addItem("(No DRS call records yet)")
        layout.addWidget(self._calls_list)
        return w

    def _current_list(self) -> QListWidget:
        idx = self._tabs.currentIndex()
        if idx == 0:
            return self._clips_list
        if idx == 1:
            return self._matches_list
        return self._calls_list

    def _open_selected(self) -> None:
        lst = self._current_list()
        item = lst.currentItem()
        if not item:
            return
        path_str = item.data(Qt.ItemDataRole.UserRole)
        if not path_str or not Path(path_str).is_file():
            QMessageBox.information(self, "Open", "Select a video file from Clips or Match Recordings.")
            return
        if self._on_open_video:
            self._on_open_video(path_str)
            self.accept()

    def _open_folder(self) -> None:
        import os
        import subprocess
        import sys

        idx = self._tabs.currentIndex()
        folder = user_clips_dir() if idx != 1 else user_matches_dir()
        if sys.platform == "win32":
            os.startfile(folder)  # noqa: S606
        else:
            subprocess.run(["xdg-open", str(folder)], check=False)
