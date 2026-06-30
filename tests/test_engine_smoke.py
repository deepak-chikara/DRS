"""DRSEngine smoke test."""

from pathlib import Path

import cv2

from drs.config import load_config
from drs.engine import DRSEngine
from synthetic_frames import make_synthetic_frame


def test_engine_processes_frames():
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "config" / "default.yaml")
    video = root / "lbw.mp4"
    if not video.is_file():
        video = root.parent / "lbw-2.mp4"
    if video.is_file():
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
    else:
        cfg.video_path = str(root / "tests" / "fixtures" / "_synthetic.mp4")
        cfg.detection_mode = "color"
        cfg.detection_scale = 1.0
        synth_path = cfg.video_path
        synth_path_obj = Path(synth_path)
        synth_path_obj.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(synth_path_obj),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (640, 360),
        )
        for _ in range(10):
            writer.write(make_synthetic_frame())
        writer.release()
        engine = DRSEngine(cfg)
        ok, err = engine.open()
        assert ok, err
        from drs.ui.playback import PlaybackState

        pb = PlaybackState()
        frame, read_ok = engine.read_frame(pb)
        assert read_ok and frame is not None
        out = engine.process_frame(frame)
        assert out is not None
        engine.close()
