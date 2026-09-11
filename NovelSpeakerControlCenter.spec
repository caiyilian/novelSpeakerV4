# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


root = Path(SPECPATH)
src = root / "src"

a = Analysis(
    [str(src / "control_center.py")],
    pathex=[str(src)],
    binaries=[],
    datas=[],
    hiddenimports=["run_label"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="NovelSpeakerControlCenter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(root / "packaging" / "NovelSpeakerControlCenter.ico"),
    version=str(root / "packaging" / "windows_version_info.txt"),
)
