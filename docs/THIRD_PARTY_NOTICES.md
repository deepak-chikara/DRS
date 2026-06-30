# Third-Party Notices

DRS Pro includes or depends on the following open-source components:

| Component | License | Notes |
|-----------|---------|-------|
| OpenCV (opencv-python) | Apache 2.0 | Video I/O and image processing |
| NumPy | BSD | Numerical computing |
| PySide6 / Qt | LGPL v3 | Desktop user interface |
| PyYAML | MIT | Configuration files |
| cvzone | MIT | Color detection helpers |
| httpx | BSD | Ollama HTTP client |
| Ultralytics YOLO | **AGPL-3.0** | Optional hybrid/yolo detection |

FFmpeg may be used indirectly by OpenCV for video decoding on some systems.

## Ultralytics YOLO (AGPL-3.0) — commercial distribution

Hybrid and YOLO detection modes load Ultralytics YOLO weights (`yolov8n.pt` by default). AGPL-3.0 requires that if you **distribute** DRS Pro (installer, SaaS, or modified binaries) you must:

1. Provide corresponding source for AGPL-covered components (Ultralytics and any modifications).
2. Include a prominent notice that the product uses AGPL software.
3. Not convey additional restrictions beyond the license.

**Options for commercial sale:**

- Ship **color-only** mode without bundling Ultralytics (fastest compliance path).
- Obtain a commercial license from Ultralytics if you need proprietary distribution with YOLO bundled.
- Offer source download / written offer per AGPL Section 13.

DRS Pro downloads YOLO weights to `%LOCALAPPDATA%\DRS\models\` on first hybrid use when not bundled. Review with legal counsel before commercial sale.

Full license texts are available from each project's repository.

Replace vendor contact details in README before commercial sale.
