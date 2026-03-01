# PR: Multi-Python, multi-CUDA precompiled wheels with object reuse

## Summary

Overhaul the wheel-building CI to produce **20 compiled wheels** covering **4 Python versions × 3 CUDA variants × 2 operating systems**, up from Python 3.10 only. Uses CUDA object reuse across Python versions to keep build time manageable — 5 CI jobs produce 20 wheels with only ~3× the compute of the original single-Python setup.

## Motivation

Previously, precompiled wheels were only published for Python 3.10. Users on 3.11, 3.12, or 3.13 had to either build from source (requiring a CUDA toolkit and C++ compiler) or rely on JIT compilation on first import. This is especially painful on Windows, where Visual Studio setup is non-trivial, and in managed environments like Azure ML, where JIT compilation is impractical.

## Key design decisions

### Per-Python wheels (not cross-Python)

Cross-Python binary sharing was extensively investigated and rejected:

- **pybind11 CPython ABI**: pybind11 uses unstable CPython internals (`_PyThreadState_UncheckedGet`, struct layouts, etc.) that change between minor versions. Wheels built with Python 3.10 segfault when loaded on 3.11+. Python 3.13 additionally removed `_PyThreadState_UncheckedGet` entirely.
- **`no_python_abi_suffix=True`** only affects the `.so` filename — the compiled code still embeds version-specific CPython struct offsets.
- **Alternatives explored**: abi3/Limited API (pybind11 doesn't support it), nanobind (can't convert `at::Tensor` without pybind11 type casters), Core DLL approach (~420 LOC, high maintenance).

**Result**: One wheel per Python version. This is the same approach PyTorch itself uses.

### Single PyTorch build version

All wheels are built against **PyTorch 2.6.0** (the latest stable release). Cross-PyTorch ABI compatibility was tested:

| Runtime PyTorch | Status | Notes |
|----------------|--------|-------|
| 2.6.0 | ✅ Works | Build version |
| 2.5.0 | ✅ Works | Backward compatible |
| 2.4.0 | ✅ Works | Backward compatible |
| 2.3.0 | ❌ Fails | `c10::SmallVectorBase::grow_pod` signature changed |

Building separate per-PyTorch wheels was deemed unnecessary since 2.6-built wheels work with 2.4+. Discovery tests in CI verify this on every run.

### CUDA object reuse

The 28 CUDA/C++ kernel source files (in `gsplat/cuda/csrc/`) are **Python-independent** — they only `#include` PyTorch headers, never Python headers. Only `ext.cpp` (the pybind11 bindings file) depends on the Python version.

This means the compiled `.o`/`.obj` files from the first Python version can be reused for subsequent versions. The build flow per CI job:

1. **Python 3.10**: Full compile (~14 min) → saves 28 `.o` files to `precompiled/`
2. **Python 3.11**: Reuses `.o` files, only recompiles `ext.cpp` (~30s)
3. **Python 3.12**: Same fast rebuild (~30s)
4. **Python 3.13**: Same fast rebuild (~30s)

This is implemented via the `GSPLAT_PRECOMPILED_OBJECTS` environment variable in `setup.py`, which passes the precompiled objects as `extra_objects` to `CUDAExtension`.

### Full CUDA version tags

Wheel names use full CUDA tags (e.g. `+pt26cu124`) matching the upstream convention. This avoids collisions between cu124 and cu126 wheels and is future-proof for new CUDA versions.

Examples: `gsplat-1.5.3+pt26cu118-cp310-cp310-linux_x86_64.whl`

### CUDA version matrix

| Index | PyTorch 2.6 Linux | PyTorch 2.6 Windows | Notes |
|-------|-------------------|---------------------|-------|
| cu118 | ✅ | ✅ (not built) | CUDA 11.8 |
| cu121 | ❌ | ❌ | **Dropped in PyTorch 2.6** |
| cu124 | ✅ | ✅ | CUDA 12.4 |
| cu126 | ✅ | ✅ | CUDA 12.6 |

PyTorch 2.6 dropped cu121 entirely — the cu121 index silently redirects to cu124 wheels, causing a toolkit/runtime mismatch. We use cu124 and cu126 for CUDA 12.x coverage.

## Wheel matrix

| OS | CUDA | Python versions | Wheels | Tag example |
|----|------|----------------|--------|-------------|
| Linux | cu118 | 3.10, 3.11, 3.12, 3.13 | 4 | `+pt26cu118` |
| Linux | cu124 | 3.10, 3.11, 3.12, 3.13 | 4 | `+pt26cu124` |
| Linux | cu126 | 3.10, 3.11, 3.12, 3.13 | 4 | `+pt26cu126` |
| Windows | cu124 | 3.10, 3.11, 3.12, 3.13 | 4 | `+pt26cu124` |
| Windows | cu126 | 3.10, 3.11, 3.12, 3.13 | 4 | `+pt26cu126` |
| | | **Total** | **20** | |

Each wheel is ~20 MB. Total release size: ~400 MB.

## CI structure

### Build jobs (5)

Each job builds 4 wheels (one per Python), taking ~16 minutes total:

| Job | OS | CUDA toolkit | Torch index | Full compile | 3 fast rebuilds |
|-----|----|-------------|-------------|-------------|-----------------|
| 1 | Linux | cu118 | cu118 | ~14 min | ~30s each |
| 2 | Linux | cu124 | cu124 | ~14 min | ~30s each |
| 3 | Linux | cu126 | cu126 | ~14 min | ~30s each |
| 4 | Windows | cu124 | cu124 | ~14 min | ~30s each |
| 5 | Windows | cu126 | cu126 | ~14 min | ~30s each |

### Test jobs (35)

- **20 same-version tests**: Each wheel tested with torch 2.6.0 from its matching CUDA index
- **15 backward compat discovery** (allow-failure): cu124 wheels tested with older PyTorch on both Linux and Windows:
  - torch 2.5.0: py3.10–3.13 Linux + py3.10–3.12 Windows = 7 jobs (no cp313 Windows wheels)
  - torch 2.4.0: py3.10–3.12 × 2 OS = 6 jobs (no cp313)
  - torch 2.3.0: py3.10 × 2 OS = 2 jobs (expected to fail — ABI break)

### Smoke test suite

Each test job runs:
1. **Symbol check** — All 28 C extension functions and 4 pybind11-registered classes present
2. **PyTorch ABI check** — `torch.version.cuda` reports expected CUDA version
3. **Public API imports** — `from gsplat import rasterization, rendering, utils`
4. **Enum access** — `CameraModelType.PINHOLE`, `ShutterType.GLOBAL`
5. **C++ class instantiation** — `UnscentedTransformParameters()`, `FThetaCameraDistortionParameters()`
6. **Shared library audit** — `ldd`/`dumpbin` to verify no rogue CUDA runtime linkage

## Full configuration test matrix

Every configuration below is one row = one testable combination. Columns:
- **Wheel**: A precompiled `.whl` is built and published
- **CI**: Smoke-tested in GitHub Actions (symbol check, import, ABI — no GPU)
- **Local**: GPU-tested by `scripts/test_wheels_local.py` (forward + backward + gradients)

### Primary configurations (PyTorch 2.6.0 — build version)

| OS | Python | CUDA | PyTorch | Wheel | CI | Local |
|----|--------|------|---------|-------|----|-------|
| Linux | 3.10 | cu118 | 2.6.0 | `gsplat-1.5.3+pt26cu118-cp310-cp310-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.11 | cu118 | 2.6.0 | `gsplat-1.5.3+pt26cu118-cp311-cp311-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.12 | cu118 | 2.6.0 | `gsplat-1.5.3+pt26cu118-cp312-cp312-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.13 | cu118 | 2.6.0 | `gsplat-1.5.3+pt26cu118-cp313-cp313-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.10 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp310-cp310-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.11 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp311-cp311-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.12 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp312-cp312-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.13 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp313-cp313-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.10 | cu126 | 2.6.0 | `gsplat-1.5.3+pt26cu126-cp310-cp310-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.11 | cu126 | 2.6.0 | `gsplat-1.5.3+pt26cu126-cp311-cp311-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.12 | cu126 | 2.6.0 | `gsplat-1.5.3+pt26cu126-cp312-cp312-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.13 | cu126 | 2.6.0 | `gsplat-1.5.3+pt26cu126-cp313-cp313-linux_x86_64.whl` | ✅ | ✅ |
| Windows | 3.10 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp310-cp310-win_amd64.whl` | ✅ | ✅ |
| Windows | 3.11 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp311-cp311-win_amd64.whl` | ✅ | ✅ |
| Windows | 3.12 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp312-cp312-win_amd64.whl` | ✅ | ✅ |
| Windows | 3.13 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp313-cp313-win_amd64.whl` | ✅ | ✅ |
| Windows | 3.10 | cu126 | 2.6.0 | `gsplat-1.5.3+pt26cu126-cp310-cp310-win_amd64.whl` | ✅ | ✅ |
| Windows | 3.11 | cu126 | 2.6.0 | `gsplat-1.5.3+pt26cu126-cp311-cp311-win_amd64.whl` | ✅ | ✅ |
| Windows | 3.12 | cu126 | 2.6.0 | `gsplat-1.5.3+pt26cu126-cp312-cp312-win_amd64.whl` | ✅ | ✅ |
| Windows | 3.13 | cu126 | 2.6.0 | `gsplat-1.5.3+pt26cu126-cp313-cp313-win_amd64.whl` | ✅ | ✅ |

### Backward compatibility (cu124 wheel + older PyTorch)

Same cu124 wheel (built for PyTorch 2.6) tested with an older PyTorch runtime.

| OS | Python | CUDA | PyTorch | Wheel | CI | Local |
|----|--------|------|---------|-------|----|-------|
| Linux | 3.10 | cu124 | 2.5.0 | `gsplat-1.5.3+pt26cu124-cp310-cp310-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.11 | cu124 | 2.5.0 | `gsplat-1.5.3+pt26cu124-cp311-cp311-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.12 | cu124 | 2.5.0 | `gsplat-1.5.3+pt26cu124-cp312-cp312-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.13 | cu124 | 2.5.0 | `gsplat-1.5.3+pt26cu124-cp313-cp313-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.10 | cu124 | 2.4.0 | `gsplat-1.5.3+pt26cu124-cp310-cp310-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.11 | cu124 | 2.4.0 | `gsplat-1.5.3+pt26cu124-cp311-cp311-linux_x86_64.whl` | ✅ | ✅ |
| Linux | 3.12 | cu124 | 2.4.0 | `gsplat-1.5.3+pt26cu124-cp312-cp312-linux_x86_64.whl` | ✅ | ✅ |
| Windows | 3.10 | cu124 | 2.5.0 | `gsplat-1.5.3+pt26cu124-cp310-cp310-win_amd64.whl` | ✅ | ✅ |
| Windows | 3.11 | cu124 | 2.5.0 | `gsplat-1.5.3+pt26cu124-cp311-cp311-win_amd64.whl` | ✅ | ✅ |
| Windows | 3.12 | cu124 | 2.5.0 | `gsplat-1.5.3+pt26cu124-cp312-cp312-win_amd64.whl` | ✅ | ✅ |
| Windows | 3.10 | cu124 | 2.4.0 | `gsplat-1.5.3+pt26cu124-cp310-cp310-win_amd64.whl` | ✅ | ✅ |
| Windows | 3.11 | cu124 | 2.4.0 | `gsplat-1.5.3+pt26cu124-cp311-cp311-win_amd64.whl` | ✅ | ✅ |
| Windows | 3.12 | cu124 | 2.4.0 | `gsplat-1.5.3+pt26cu124-cp312-cp312-win_amd64.whl` | ✅ | ✅ |

Same wheels as in the primary table — no separate wheel built for older PyTorch.

Note: PyTorch 2.4.0 caps at Python 3.12 (no cp313 wheels on either OS). PyTorch 2.5.0 has cp313 on Linux but not Windows. All backward compat CI tests use `continue-on-error: true` so they don't block the build — but they pass consistently.

### Unsupported configurations

| Configuration | Reason |
|---------------|--------|
| Windows + cu118 | Not built (could be added — PyTorch 2.6 has cu118 Windows wheels) |
| Any OS + cu121 | Dropped by PyTorch 2.6 (silently redirects to cu124) |
| Any OS + PyTorch 2.3.0 | `c10::SmallVectorBase::grow_pod` signature changed |
| Python 3.14 | PyTorch doesn't ship cp314 wheels yet |
| macOS + any CUDA | No CUDA support on macOS |

### Coverage gaps

All supported configurations are now CI-tested across all Python versions. The only remaining gap is **GPU-level testing** — CI runners lack GPUs, so kernel correctness (forward pass, backward pass, gradients) is only verified by `test_wheels_local.py` on a local GPU machine.

## Changes

### New files

- **`.github/workflows/build_one_wheel.sh`** (~180 lines) — Per-Python build helper. Handles torch install, Windows patches (MSVC compatibility, missing `.lib` generation, `cuda_cmake_macros.h` stub), version stamping, compile/reuse, and smoke test.

- **`docs/INSTALL_PRECOMPILED.md`** — Installation guide for precompiled wheels with version compatibility table and troubleshooting.

- **`scripts/test_wheels_local.py`** — Standalone GPU test script using `uv` for ephemeral venvs. Tests the full matrix locally without conda.

- **`.github/workflows/retag_wheel.py`** — Wheel retagging utility (retained from earlier exploration, may be useful for future multi-Python tag experiments).

- **`.github/workflows/find_ext.py`** — Helper to locate compiled extension without importing (avoids CUDA init on non-GPU runners).

### Modified files

- **`setup.py`** — Added `GSPLAT_PRECOMPILED_OBJECTS` support: when set, only compiles `ext.cpp` and links precompiled `.o`/`.obj` files. Also fixes CUDA_HOME cache stale-read issue and Windows `.lib` filtering for `extra_objects`.

- **`.github/workflows/building.yml`** — Complete rewrite: 5-job build matrix with object reuse, 35-job test matrix, backward compat discovery.

- **`.github/workflows/cuda/{Linux,Windows}.sh`** — Added cu126 case (CUDA 12.6.3).
- **`.github/workflows/cuda/{Linux,Windows}-env.sh`** — Added cu126 environment variables.

- **`.github/workflows/publish.yml`** — Updated artifact download pattern, uses `${{ github.repository }}` for fork compatibility.

- **`.github/workflows/generate_simple_index_pages.py`** — Updated regex for compressed wheel tags.

- **`README.md`** — Updated precompiled wheel section for Python 3.10–3.13.

- **`docs/INSTALL_WIN.md`** — Links to `INSTALL_PRECOMPILED.md`.

## Windows build fixes

Six issues were fixed to enable Windows wheel building with PyTorch 2.6:

1. **CUDA_HOME stale cache** — PyTorch's `_find_cuda_home()` caches `CUDA_HOME` at import time. `setup.py` now force-refreshes `torch.utils.cpp_extension.CUDA_HOME` from the environment.

2. **Missing `cuda_cmake_macros.h`** — PyTorch 2.6 pip wheels on Windows omit this header. `build_one_wheel.sh` creates a stub.

3. **Missing `.lib` files** — `c10_cuda.lib` and `torch_cuda.lib` don't exist in PyTorch pip wheels on Windows. The build script generates them from DLLs using `dumpbin` + `lib.exe`.

4. **CPU-only torch from cu121** — The cu121 PyTorch index has no CUDA wheels for Windows (only CPU). All Windows builds use cu124+ indexes.

5. **PyTorch `Parallel.h` patch** — `static constexpr` in PyTorch headers is incompatible with MSVC on `windows-2022` runners. Patched in `build_one_wheel.sh`.

6. **`.lib` in `extra_objects`** — `setup.py` filters out `.lib` files from `extra_objects` on Windows (CUDAExtension doesn't accept them as extra objects).

## Limitations and future work

- **No GPU CI tests**: CI runners lack GPUs. Smoke tests verify binary compatibility but not kernel correctness. Use `scripts/test_wheels_local.py` for GPU validation.
- **Python 3.14**: Not included — PyTorch doesn't yet ship cp314 wheels.
- **PyTorch 2.4/2.5 wheels**: Not built separately. Wheels built with 2.6 are runtime-compatible with 2.4+ (verified in CI). Separate 2.4/2.5 wheels could be added as additional build jobs if needed.
- **Windows cu118**: Not built (could be added — PyTorch 2.6 has cu118 Windows wheels).
- **macOS**: No CUDA support.
