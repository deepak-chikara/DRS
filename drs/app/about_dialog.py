"""About dialog."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from drs import __app_name__, __version__


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {__app_name__}")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<h2>{__app_name__}</h2>"))
        layout.addWidget(QLabel(f"Version {__version__}"))
        layout.addWidget(QLabel(
            "LBW decision review assist for club cricket.<br><br>"
            "<b>What DRS Pro checks:</b> ball position on the pad relative to the "
            "stump corridor (off-stump to leg-stump lines) from your single camera view.<br><br>"
            "<b>What it does NOT check:</b> ball height above the stumps, bat before pad, "
            "pitching outside off, snick, or deflections. Side elevation is a height hint only — "
            "not an automatic verdict.<br><br>"
            "<b>Disclaimer:</b> Line assist for club review — not an official umpire decision "
            "and not broadcast Hawk-Eye."
        ))
        layout.addWidget(QLabel("© 2025 DRS. All rights reserved."))
