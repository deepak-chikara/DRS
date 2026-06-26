"""About dialog."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from drs import __app_name__, __version__


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {__app_name__}")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<h2>{__app_name__}</h2>"))
        layout.addWidget(QLabel(f"Version {__version__}"))
        layout.addWidget(QLabel(
            "LBW decision review assist for club cricket.<br><br>"
            "<b>Disclaimer:</b> DRS provides LBW assistance only. "
            "It does not make official umpiring decisions."
        ))
        layout.addWidget(QLabel("© 2025 DRS. All rights reserved."))
