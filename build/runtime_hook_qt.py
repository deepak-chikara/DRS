"""PyInstaller runtime hook — Qt plugins and bundle paths."""

import os
import sys

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    base = sys._MEIPASS
    for sub in ("PySide6/plugins", "PySide6/Qt/plugins"):
        plugin_path = os.path.join(base, sub.replace("/", os.sep))
        if os.path.isdir(plugin_path):
            os.environ["QT_PLUGIN_PATH"] = plugin_path
            break
