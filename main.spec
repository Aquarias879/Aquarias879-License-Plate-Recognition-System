# Import necessary modules
import os
from PyInstaller.utils.hooks import collect_submodules

# Collect hidden imports
hidden_imports = ['onnxruntime', 'openvino']

# Add ONNXRuntime .dll and .pyd files
onnx_files = [
    ('onnx/lib/site-packages/onnxruntime/capi/*.dll', 'onnxruntime/capi'),
    ('onnx/lib/site-packages/onnxruntime/capi/*.pyd', 'onnxruntime/capi')
]

# Data files including static files and ONNXRuntime binaries
data_files = [
    ('static', 'static')
] + onnx_files

# Analysis step
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=data_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False
)

# Create Python module archive
pyz = PYZ(a.pure)

# Create executable with bootloader debug and UPX compression
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ALPR',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to False if you don't want a console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='static/icon/cctv.ico'  # Icon for the application
)

# Create UPX compression
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ALPR'
)
