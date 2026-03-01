#!/bin/bash
# Build a single gsplat wheel for a given Python version.
#
# Usage:
#   bash build_one_wheel.sh <py-version> <torch-version> <default-torch-cuda> <build-cuda> [precompiled-dir]
#
# Arguments:
#   py-version          Python version (e.g. 3.10)
#   torch-version       PyTorch version (e.g. 2.6.0)
#   default-torch-cuda  Torch CUDA index to install from (e.g. cu118, cu124)
#   build-cuda          CUDA toolkit used for compilation (e.g. cu118, cu124)
#   precompiled-dir     Optional: directory of pre-compiled .o/.obj files to reuse
#
# Environment:
#   RUNNER_OS           Set by GitHub Actions (Linux / Windows)
#   CUDA_HOME           Must be set before calling
#
# Outputs:
#   Wheel in dist/py<version>/*.whl
#   (first build only) precompiled/ directory with .o/.obj files
set -e

PYVER="$1"
TORCH_VER="$2"
DEFAULT_TORCH_CUDA="$3"
BUILD_CUDA="$4"
PRECOMPILED="$5"

echo "================================================================"
echo "Building wheel: Python $PYVER, torch $TORCH_VER, CUDA $BUILD_CUDA"
[ -n "$PRECOMPILED" ] && echo "  Reusing pre-compiled objects from: $PRECOMPILED"
echo "================================================================"

# --- Install dependencies ---
# Starting from PyTorch 2.6, wheel names on the CUDA indexes no longer
# include a +cuXYZ local version suffix (the wheel IS CUDA-enabled, but
# pip reports it as plain 2.6.0). Install without the local tag.
TORCH_CUDA="$DEFAULT_TORCH_CUDA"
pip install torch==${TORCH_VER} \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
    --extra-index-url https://pypi.org/simple
pip install ninja wheel setuptools

# Verify CUDA torch
python -c "
import torch
cuda_ver = torch.version.cuda
assert cuda_ver is not None, (
    f'CPU-only PyTorch was installed ({torch.__version__}). '
    'Expected CUDA variant with torch.version.cuda set.'
)
print(f'PyTorch {torch.__version__}, CUDA {cuda_ver}')
"

# --- Windows-specific patches ---
if [ "$RUNNER_OS" = "Windows" ]; then
    # Patch Parallel.h (static constexpr issue with MSVC)
    python -c "
import re, torch, pathlib
ph = pathlib.Path(torch.__file__).parent / 'include' / 'ATen' / 'Parallel.h'
text = ph.read_text()
new = re.sub(
    r'(?:(?:inline|static|constexpr|TORCH_API)\s+)*void\s+lazy_init_num_threads\s*\(\s*\)\s*\{(?:[^{}]*|\{[^{}]*\})*\}',
    'TORCH_API void lazy_init_num_threads();',
    text, count=1)
if new != text:
    ph.write_text(new)
    print('Patched Parallel.h')
else:
    print('No patching needed for Parallel.h')
"

    # Create stub cuda_cmake_macros.h if missing
    python -c "
import torch, pathlib
inc = pathlib.Path(torch.__file__).parent / 'include'
target = inc / 'c10' / 'cuda' / 'impl' / 'cuda_cmake_macros.h'
if not target.exists():
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('#pragma once\n#define C10_CUDA_BUILD_SHARED_LIBS\n')
    print(f'Created stub: {target}')
else:
    print(f'Already exists: {target}')
"

    # Generate missing .lib import libraries
    python -c "
import torch, pathlib, subprocess, sys, os, glob as glob_mod
lib_dir = pathlib.Path(torch.__file__).parent / 'lib'
lib_exe = 'lib'
dumpbin_exe = 'dumpbin'
msvc_bins = glob_mod.glob(
    r'C:\Program Files*\Microsoft Visual Studio\*\*'
    r'\VC\Tools\MSVC\*\bin\Hostx64\x64')
if msvc_bins:
    msvc_bin = msvc_bins[0]
    lib_exe = os.path.join(msvc_bin, 'lib.exe')
    dumpbin_exe = os.path.join(msvc_bin, 'dumpbin.exe')
needed = ['c10', 'c10_cuda', 'torch', 'torch_cpu', 'torch_cuda', 'torch_python']
for name in needed:
    lib = lib_dir / f'{name}.lib'
    dll = lib_dir / f'{name}.dll'
    if lib.exists():
        continue
    if dll.exists():
        deffile = lib_dir / f'{name}.def'
        result = subprocess.run(
            [dumpbin_exe, '/EXPORTS', str(dll)],
            capture_output=True, text=True)
        lines = result.stdout.splitlines()
        exports = []
        in_table = False
        for line in lines:
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                in_table = True
                exports.append(parts[3])
            elif in_table and not line.strip():
                break
        with open(deffile, 'w') as f:
            f.write(f'LIBRARY {name}\nEXPORTS\n')
            for exp in exports:
                f.write(f'  {exp}\n')
        subprocess.run(
            [lib_exe, f'/DEF:{deffile}', f'/OUT:{lib}', '/MACHINE:X64'],
            check=True)
        print(f'  Created {lib.name} from {dll.name}')
    else:
        deffile = lib_dir / f'{name}.def'
        with open(deffile, 'w') as f:
            f.write(f'LIBRARY {name}\nEXPORTS\n')
        subprocess.run(
            [lib_exe, f'/DEF:{deffile}', f'/OUT:{lib}', '/MACHINE:X64'],
            check=True)
        print(f'  Created empty stub {lib.name}')
" || true
fi

# --- Set version string ---
# Use full CUDA tag (e.g. cu118, cu124) to match upstream convention
# and avoid collisions when multiple CUDA 12.x versions coexist.
VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' gsplat/version.py)
TORCH_TAG=$(echo "pt${TORCH_VER}" | sed 's/..$//' | sed 's/\.//g')
LOCAL="${TORCH_TAG}${BUILD_CUDA}"
sed -i "s/$VERSION/$VERSION+$LOCAL/" gsplat/version.py
echo "Version: $VERSION+$LOCAL"

# --- Build wheel ---
DIST_DIR="dist/py${PYVER}"
mkdir -p "$DIST_DIR"

# Clean previous build artifacts (keep precompiled objects dir untouched)
rm -rf build/lib.* build/bdist.* *.egg-info gsplat/*.egg-info

if [ -n "$PRECOMPILED" ] && [ -d "$PRECOMPILED" ]; then
    echo "=== Fast rebuild: reusing $(ls "$PRECOMPILED"/*.o "$PRECOMPILED"/*.obj 2>/dev/null | wc -l) pre-compiled objects ==="
    GSPLAT_PRECOMPILED_OBJECTS="$PRECOMPILED" MAX_JOBS=2 python setup.py bdist_wheel --dist-dir="$DIST_DIR"
else
    echo "=== Full compile ==="
    MAX_JOBS=2 python setup.py bdist_wheel --dist-dir="$DIST_DIR"

    # Save compiled objects for subsequent Python versions
    echo "=== Collecting pre-compiled objects ==="
    mkdir -p precompiled
    find build/temp.* \( -name "*.o" -o -name "*.obj" \) \
        ! -name "ext.o" ! -name "ext.obj" \
        -exec cp {} precompiled/ \;
    echo "  Collected $(ls precompiled/ | wc -l) objects"
fi

# --- Reset version for next build ---
git checkout gsplat/version.py

# --- Quick smoke test ---
pip install "$DIST_DIR"/*.whl
python -c "import gsplat; print(f'gsplat {gsplat.__version__} OK')"

echo "=== Done: Python $PYVER ==="
ls -la "$DIST_DIR/"
