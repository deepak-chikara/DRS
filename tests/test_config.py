"""Config loading tests."""

from pathlib import Path

from drs.config import load_config


def test_load_default_config():
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "config" / "default.yaml")
    assert cfg.mode == "file"
    assert cfg.ground_id == "default"
    assert "lbw" in cfg.video_path.lower() or cfg.video_path.endswith(".mp4")
