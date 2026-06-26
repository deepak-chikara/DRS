# DRS Pro

**Professional LBW decision review assist for club cricket.**

DRS Pro helps umpires and clubs review deliveries with stump corridor overlays, ball and batsman tracking, and OUT/NOT OUT assistance. It is an **aid for human review**, not an official umpiring system.

## System requirements

- Windows 10 or 11 (64-bit)
- 8 GB RAM minimum (16 GB recommended for live camera + YOLO)
- 1920×1080 display recommended
- ~500 MB disk space after install

## Install (customers)

1. Run `DRS-Pro-Setup-1.0.0.exe` from your vendor.
2. Follow the first-run setup wizard to calibrate stump lines for your ground.
3. Open a match video via **File → Open Video**.

User data is stored in `%LOCALAPPDATA%\DRS` (settings, calibration, clips, logs).

## Install (developers)

```powershell
cd DRS
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
python main.py
```

## Build installer

```powershell
pip install pyinstaller
pyinstaller build/drs.spec
# Then compile build/installer.iss with Inno Setup 6
```

Output: `dist/installer/DRS-Pro-Setup-1.0.0.exe`

## Documentation

- [User Guide](docs/USER_GUIDE.md) — controls, calibration, tuning
- [EULA](docs/EULA.md) — end-user license
- [Privacy](docs/PRIVACY.md) — local data only
- [Third-party notices](docs/THIRD_PARTY_NOTICES.md)

## Support

Contact: **support@yourcompany.com** (replace before sale)

## License

Source repository: MIT (see LICENSE).  
Customers receive software under the [EULA](docs/EULA.md).
