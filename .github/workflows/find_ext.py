#!/usr/bin/env python3
"""Locate the compiled gsplat C extension (.so / .pyd) without importing it.

Importing gsplat.csrc directly would trigger CUDA runtime initialization,
which fails on CI runners without a GPU.  Instead we find the file via a
filesystem glob inside the installed gsplat package directory.

Usage (from a shell where gsplat is installed):
    EXT=$(python .github/workflows/find_ext.py)
"""

import pathlib
import sys

import gsplat

pkg = pathlib.Path(gsplat.__file__).parent
matches = sorted(pkg.glob("csrc*"))
if not matches:
    print(f"ERROR: csrc extension not found in {pkg}", file=sys.stderr)
    sys.exit(1)
print(matches[0])
