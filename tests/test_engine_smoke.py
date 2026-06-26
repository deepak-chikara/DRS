"""DRSEngine smoke test."""

from pathlib import Path

from drs.config import load_config
from drs.engine import DRSEngine


def test_engine_processes_frames():
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "config" / "default.yaml")
    video = root / "lbw.mp4"
    if not video.is_file():
        video = root.parent / "lbw-2.mp4"
    if not video.is_file():
        return  # skip if no sample video in CI
    cfg.video_path = str(video)
    engine = DRSEngine(cfg)
    ok, err = engine.open()
    assert ok, err
    from drs.ui.playback import PlaybackState

    pb = PlaybackState()
    for _ in range(5):
        frame, read_ok = engine.read_frame(pb)
        if not read_ok:
            break
        out = engine.process_frame(frame)
        assert out is not None
        assert out.shape == frame.shape
        pb.frame_pos += 1
    engine.close()
