# Build DRS Pro — complete Windows bundle (PyInstaller + optional Inno Setup installer)
# Output:
#   dist/DRS Pro/DRS Pro.exe     — portable folder (all dependencies included)
#   dist/installer/DRS-Pro-Setup-1.0.0.exe — one-click installer (if Inno Setup installed)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== DRS Pro build ===" -ForegroundColor Cyan

# 1. Icon (multi-size for Windows taskbar / shortcuts)
python tools/generate_icon.py

# 2. Ensure build tools and runtime deps
python -m pip install --quiet pyinstaller -r requirements.txt 2>$null

# 3. PyInstaller one-folder bundle
Write-Host "Running PyInstaller (this may take several minutes)..." -ForegroundColor Yellow
python -m PyInstaller build/drs.spec --noconfirm --clean

$bundleDir = Join-Path (Get-Location) "dist\DRS Pro"
$exe = Join-Path $bundleDir "DRS Pro.exe"
$iconSrc = Join-Path (Get-Location) "assets\icon.ico"

if (-not (Test-Path $exe)) {
    Write-Error "Build failed: $exe not found"
}

# 4. Copy icon beside exe for shortcuts / Qt fallback
Copy-Item -Force $iconSrc (Join-Path $bundleDir "icon.ico")

Write-Host ""
Write-Host "Portable bundle ready:" -ForegroundColor Green
Write-Host "  $exe"
Write-Host "  Run directly or zip the 'dist\DRS Pro' folder to distribute."

# 5. Optional installer
$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if (-not $iscc) {
    $defaultIscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (Test-Path $defaultIscc) { $iscc = $defaultIscc }
}

if ($iscc) {
    Write-Host "Building installer with Inno Setup..." -ForegroundColor Yellow
    & $iscc build/installer.iss
    $setup = Join-Path (Get-Location) "dist\installer\DRS-Pro-Setup-1.0.0.exe"
    if (Test-Path $setup) {
        Write-Host "Installer ready:" -ForegroundColor Green
        Write-Host "  $setup"
    }
} else {
    Write-Host ""
    Write-Host "Inno Setup not found — skipping setup EXE." -ForegroundColor Yellow
    Write-Host "Install Inno Setup 6 from https://jrsoftware.org/isinfo.php then re-run this script,"
    Write-Host "or distribute the portable folder: dist\DRS Pro"
}

Write-Host ""
Write-Host "Done." -ForegroundColor Cyan
