"""DRS Pro desktop application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drs import __app_name__, __version__
from drs.app.main_window import MainWindow
from drs.app.setup_wizard import SetupWizard
from drs.config import load_user_config
from drs.logging_setup import setup_logging
from drs.paths import app_root


def _load_theme(app: QApplication) -> None:
    theme = app_root() / "drs" / "app" / "theme.qss"
    if theme.is_file():
        app.setStyleSheet(theme.read_text(encoding="utf-8"))


def run(config=None) -> int:
    setup_logging()
    app = QApplication(sys.argv)
    app.setApplicationName(__app_name__)
    app.setApplicationVersion(__version__)
    _load_theme(app)

    try:
        if config is None:
            config = load_user_config()
    except FileNotFoundError as exc:
        QMessageBox.critical(None, "DRS Pro", f"Configuration error:\n{exc}")
        return 1

    if MainWindow.needs_setup(config):
        reply = QMessageBox.question(
            None,
            "First-time setup",
            "No stump calibration found for this ground.\n\nRun setup wizard now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            wizard = SetupWizard(config)
            if not wizard.run_calibration():
                QMessageBox.warning(None, "Setup", "Setup skipped — heuristic stump lines will be used.")

    window = MainWindow(config)
    window.show()
    if config.video_path and Path(config.video_path).is_file():
        window._start_worker()
        window._status.showMessage(f"Loaded: {Path(config.video_path).name}")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
