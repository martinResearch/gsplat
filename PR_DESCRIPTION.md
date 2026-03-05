# PR: Multi-Python, multi-CUDA precompiled wheels with object reuse

## Summary

This PR overhauls the wheel-building CI to produce **28 precompiled wheels** that cover **4 Python versions (3.10–3.13) × 4 CUDA variants × 2 operating systems (Linux and Windows)**. Previously, wheels were only published for Python 3.10. To keep build times manageable, compiled CUDA objects are reused across Python versions — so just 7 CI build jobs are enough to produce all 28 wheels, using only about 3× the compute of the original single-Python setup.

## Motivation

Before this change, precompiled wheels were only available for Python 3.10. Users running Python 3.11, 3.12, or 3.13 had two options: build from source (which requires a CUDA toolkit and a C++ compiler) or rely on JIT compilation on first import. Both options are especially painful on Windows, where setting up Visual Studio and the CUDA toolkit is non-trivial, and in managed environments like Azure ML, where JIT compilation may not be practical at all.

## Key design decisions

### Per-Python wheels (not cross-Python)

We investigated whether a single compiled binary could be shared across Python versions, but this is not possible:

- **pybind11 depends on CPython internals**: pybind11 uses unstable CPython internals like `_PyThreadState_UncheckedGet` and version-specific struct layouts. These change between minor Python versions, so a wheel built with Python 3.10 will segfault when loaded on 3.11 or later. Python 3.13 removed `_PyThreadState_UncheckedGet` entirely.
- **`no_python_abi_suffix=True` doesn't help**: This flag only changes the `.so` filename. The compiled code still contains version-specific CPython struct offsets baked in at compile time.
- **Other approaches were explored and ruled out**: The Python stable ABI (`abi3` / Limited API) is not supported by pybind11. Nanobind can't convert `at::Tensor` without pybind11 type casters. A "Core DLL" approach (putting CUDA kernels in a shared library) would work but adds ~420 lines of build code and ongoing maintenance burden.

As a result, we build one wheel per Python version. This is the same approach that PyTorch itself uses.

### PyTorch versions per CUDA index

Each CUDA variant uses the **latest available PyTorch** on its PyTorch index:

| CUDA | PyTorch | Tag |
|------|---------|-----|
| cu118 | 2.7.1 | pt27 |
| cu124 | 2.6.0 | pt26 |
| cu126 | 2.10.0 | pt210 |
| cu128 | 2.10.0 | pt210 |

Backward compatibility with older PyTorch versions is **not supported** — C++ ABI symbols change between PyTorch minor releases, so wheels built for one PyTorch version will not work correctly with a different version. On Windows, this causes an immediate import failure. On Linux, the import may appear to succeed but the code will likely crash on the first GPU call.

This is in contrast with our support for multiple Python and CUDA versions:

- **Multiple Python versions (3.10–3.13)** are worth supporting because users are often locked to a specific Python version by other dependencies in their environment. The cost is also low: thanks to CUDA object reuse (see [Build jobs](#build-jobs-7)), each additional Python version only adds ~30 seconds of compile time per build job.
- **Multiple CUDA versions (cu118, cu124, cu126, cu128)** are worth supporting because users' GPU drivers dictate which CUDA versions they can run. Upgrading the driver is not always possible — for example on shared clusters or managed environments. Each CUDA version does require a full separate build (~14 min), but four versions cover the range of commonly deployed hardware.

### Full CUDA version tags

Wheel filenames include both a PyTorch and CUDA version tag (e.g. `+pt27cu118`) that follows the same convention used by PyTorch. This prevents collisions between wheels built for different CUDA or PyTorch versions.

For example: `gsplat-1.5.3+pt27cu118-cp310-cp310-linux_x86_64.whl`

### CUDA version matrix

| Index | Linux | Windows | PyTorch | Notes |
|-------|-------|---------|---------|-------|
| cu118 | ✅ | ❌ (not built) | 2.7.1 | CUDA 11.8, Linux-only |
| cu121 | ❌ | ❌ | — | **Dropped in PyTorch 2.6+** |
| cu124 | ✅ | ✅ | 2.6.0 | CUDA 12.4 (latest torch on cu124 index) |
| cu126 | ✅ | ✅ | 2.10.0 | CUDA 12.6 |
| cu128 | ✅ | ✅ | 2.10.0 | CUDA 12.8 |

PyTorch 2.6+ dropped cu121 entirely. The cu121 package index silently redirects to cu124 wheels, which would cause a toolkit/runtime mismatch. PyTorch 2.7+ dropped cu124, so cu124 wheels use the latest available torch 2.6.0.

## Wheel matrix

The table below lists all 28 wheels that are built and published. Each row is one wheel. The **CI** column indicates whether the wheel is smoke-tested in GitHub Actions (symbol check, imports, ABI — no GPU). The **Local** column indicates whether it has been GPU-tested locally (forward pass, backward pass, and gradient checks).

| OS | Python | CUDA | PyTorch | Wheel | CI | Local |
|----|--------|------|---------|-------|----|-------|
| Linux | 3.10 | cu118 | 2.7.1 | `gsplat-1.5.3+pt27cu118-cp310-cp310-linux_x86_64.whl` | ✅ | — |
| Linux | 3.11 | cu118 | 2.7.1 | `gsplat-1.5.3+pt27cu118-cp311-cp311-linux_x86_64.whl` | ✅ | — |
| Linux | 3.12 | cu118 | 2.7.1 | `gsplat-1.5.3+pt27cu118-cp312-cp312-linux_x86_64.whl` | ✅ | — |
| Linux | 3.13 | cu118 | 2.7.1 | `gsplat-1.5.3+pt27cu118-cp313-cp313-linux_x86_64.whl` | ✅ | — |
| Linux | 3.10 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp310-cp310-linux_x86_64.whl` | ✅ | — |
| Linux | 3.11 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp311-cp311-linux_x86_64.whl` | ✅ | — |
| Linux | 3.12 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp312-cp312-linux_x86_64.whl` | ✅ | — |
| Linux | 3.13 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp313-cp313-linux_x86_64.whl` | ✅ | — |
| Linux | 3.10 | cu126 | 2.10.0 | `gsplat-1.5.3+pt210cu126-cp310-cp310-linux_x86_64.whl` | ✅ | — |
| Linux | 3.11 | cu126 | 2.10.0 | `gsplat-1.5.3+pt210cu126-cp311-cp311-linux_x86_64.whl` | ✅ | — |
| Linux | 3.12 | cu126 | 2.10.0 | `gsplat-1.5.3+pt210cu126-cp312-cp312-linux_x86_64.whl` | ✅ | — |
| Linux | 3.13 | cu126 | 2.10.0 | `gsplat-1.5.3+pt210cu126-cp313-cp313-linux_x86_64.whl` | ✅ | — |
| Linux | 3.10 | cu128 | 2.10.0 | `gsplat-1.5.3+pt210cu128-cp310-cp310-linux_x86_64.whl` | ✅ | — |
| Linux | 3.11 | cu128 | 2.10.0 | `gsplat-1.5.3+pt210cu128-cp311-cp311-linux_x86_64.whl` | ✅ | — |
| Linux | 3.12 | cu128 | 2.10.0 | `gsplat-1.5.3+pt210cu128-cp312-cp312-linux_x86_64.whl` | ✅ | — |
| Linux | 3.13 | cu128 | 2.10.0 | `gsplat-1.5.3+pt210cu128-cp313-cp313-linux_x86_64.whl` | ✅ | — |
| Windows | 3.10 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp310-cp310-win_amd64.whl` | ✅ | — |
| Windows | 3.11 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp311-cp311-win_amd64.whl` | ✅ | — |
| Windows | 3.12 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp312-cp312-win_amd64.whl` | ✅ | — |
| Windows | 3.13 | cu124 | 2.6.0 | `gsplat-1.5.3+pt26cu124-cp313-cp313-win_amd64.whl` | ✅ | — |
| Windows | 3.10 | cu126 | 2.10.0 | `gsplat-1.5.3+pt210cu126-cp310-cp310-win_amd64.whl` | ✅ | — |
| Windows | 3.11 | cu126 | 2.10.0 | `gsplat-1.5.3+pt210cu126-cp311-cp311-win_amd64.whl` | ✅ | — |
| Windows | 3.12 | cu126 | 2.10.0 | `gsplat-1.5.3+pt210cu126-cp312-cp312-win_amd64.whl` | ✅ | — |
| Windows | 3.13 | cu126 | 2.10.0 | `gsplat-1.5.3+pt210cu126-cp313-cp313-win_amd64.whl` | ✅ | — |
| Windows | 3.10 | cu128 | 2.10.0 | `gsplat-1.5.3+pt210cu128-cp310-cp310-win_amd64.whl` | ✅ | — |
| Windows | 3.11 | cu128 | 2.10.0 | `gsplat-1.5.3+pt210cu128-cp311-cp311-win_amd64.whl` | ✅ | — |
| Windows | 3.12 | cu128 | 2.10.0 | `gsplat-1.5.3+pt210cu128-cp312-cp312-win_amd64.whl` | ✅ | — |
| Windows | 3.13 | cu128 | 2.10.0 | `gsplat-1.5.3+pt210cu128-cp313-cp313-win_amd64.whl` | ✅ | — |

Each CUDA variant uses the latest available PyTorch on its index. Each wheel is ~20 MB. Total release size: ~560 MB.

Legend: ✅ = tested and passed, — = not tested

### Unsupported configurations

| Configuration | Reason |
|---------------|--------|
| Windows + cu118 | Not currently built. Could be added in the future since PyTorch does provide cu118 Windows wheels. |
| Any OS + cu121 | Dropped by PyTorch 2.6+. The cu121 index silently redirects to cu124, which causes a mismatch. |
| Mismatched PyTorch version | C++ ABI symbols change between PyTorch minor releases. A wheel built for one version will not work with another. |
| Python 3.14 | PyTorch does not ship cp314 wheels yet. |
| macOS + any CUDA | macOS does not support CUDA. |

### Coverage gaps

All 28 configurations (7 build jobs × 4 Python versions) pass the CI smoke tests on both Linux and Windows. However, there is a gap in GPU-level testing:

- **Linux**: No local Linux GPU test machine was available, so all 16 Linux configurations have not been GPU-tested. They pass the CI smoke tests (symbol check, imports, ABI) but have not been validated with actual CUDA kernel execution.
- **Windows cu128**: The 4 Windows cu128 configurations have not been GPU-tested locally yet (CI smoke tests pass).

The 8 Windows configurations (cu124 and cu126, each with Python 3.10–3.13) have been fully GPU-validated on an RTX 4060, including forward pass, backward pass, and gradient checks.

## CI structure

### Build jobs (7)

Each build job produces 4 wheels (one per Python version) in ~16 minutes total, down from ~56 minutes without optimization — a **3.5× reduction**. This is possible because 27 of the 28 CUDA/C++ source files never include Python headers, so their compiled object files are identical across Python versions. Only `ext.cpp` (the pybind11 bindings) needs to be recompiled for each Python version.

In practice, the first Python version does a full compile (~14 min) and saves the object files. The remaining three Python versions reuse those files and only recompile `ext.cpp` (~30s each). This is controlled by the `GSPLAT_PRECOMPILED_OBJECTS` environment variable in `setup.py`.

| Job | OS | CUDA toolkit | Torch index | Full compile | 3 fast rebuilds |
|-----|----|-------------|-------------|-------------|------------------|
| 1 | Linux | cu118 | cu118 | ~14 min | ~30s each |
| 2 | Linux | cu124 | cu124 | ~14 min | ~30s each |
| 3 | Linux | cu126 | cu126 | ~14 min | ~30s each |
| 4 | Linux | cu128 | cu128 | ~14 min | ~30s each |
| 5 | Windows | cu124 | cu124 | ~14 min | ~30s each |
| 6 | Windows | cu126 | cu126 | ~14 min | ~30s each |
| 7 | Windows | cu128 | cu128 | ~14 min | ~30s each |

### Test jobs (28)

Each of the 28 wheels is tested in its own CI job using the matching PyTorch version installed from the matching CUDA index (see [PyTorch versions per CUDA index](#pytorch-versions-per-cuda-index)).

### Smoke test suite

Each test job runs the following checks (without requiring a GPU):

1. **Symbol check** — Verifies that all 28 C extension functions and 4 pybind11-registered classes are present in the compiled module.
2. **PyTorch ABI check** — Confirms that `torch.version.cuda` reports the expected CUDA version.
3. **Public API imports** — Imports the main public modules: `from gsplat import rasterization, rendering, utils`.
4. **Enum access** — Accesses enum values like `CameraModelType.PINHOLE` and `ShutterType.GLOBAL` to verify they are properly exported.
5. **C++ class instantiation** — Creates instances of `UnscentedTransformParameters()` and `FThetaCameraDistortionParameters()` to verify pybind11 bindings work.
6. **Shared library audit** — Runs `ldd` (Linux) or `dumpbin` (Windows) to confirm the extension does not accidentally link against the CUDA runtime.

## Changes

### New files

- **`.github/workflows/build_one_wheel.sh`** (~180 lines) — A per-Python build helper script. It handles installing the correct PyTorch version, applying Windows-specific patches (MSVC compatibility, generating missing `.lib` files, creating a `cuda_cmake_macros.h` stub), stamping the version into the package, compiling or reusing precompiled objects, and running the smoke test.

- **`docs/INSTALL_PRECOMPILED.md`** — An installation guide for precompiled wheels, including a version compatibility table and troubleshooting steps.

- **`scripts/test_wheels_local.py`** — A standalone GPU test script that uses `uv` to create ephemeral virtual environments. It tests the full wheel matrix locally without requiring conda.

- **`.github/workflows/retag_wheel.py`** — A wheel retagging utility, retained from earlier exploration of cross-Python wheel sharing. May be useful for future experiments.

- **`.github/workflows/find_ext.py`** — A helper script that locates the compiled C extension without importing it. This avoids triggering CUDA initialization on CI runners that do not have GPUs.

### Modified files

- **`setup.py`** — Added support for the `GSPLAT_PRECOMPILED_OBJECTS` environment variable. When set, setup.py only compiles `ext.cpp` and links against precompiled `.o`/`.obj` files instead of compiling everything from source. Also fixes a stale-cache bug with `CUDA_HOME` detection and filters out `.lib` files from `extra_objects` on Windows.

- **`.github/workflows/building.yml`** — Completely rewritten to implement the 7-job build matrix with object reuse and the 28-job test matrix.

- **`.github/workflows/cuda/{Linux,Windows}.sh`** — Added cases for cu126 (CUDA 12.6.3) and cu128 (CUDA 12.8.1). The cu128 Windows installer was updated from 12.8.0 (570.86) to 12.8.1 (572.61) because the 12.8.0 installer is incompatible with the `windows-2022` GitHub runner image.
- **`.github/workflows/cuda/{Linux,Windows}-env.sh`** — Added environment variables for cu126 and cu128.

- **`.github/workflows/publish.yml`** — Updated the artifact download pattern and now uses `${{ github.repository }}` for compatibility with forks.

- **`.github/workflows/generate_simple_index_pages.py`** — Updated the regex to handle the new compressed wheel tag format.

- **`README.md`** — Updated the precompiled wheel section to reflect support for Python 3.10–3.13.

- **`docs/INSTALL_WIN.md`** — Added a link to `INSTALL_PRECOMPILED.md`.

## Windows build fixes

Nine issues had to be fixed to make Windows wheel building work:

1. **CUDA_HOME stale cache** — PyTorch's `_find_cuda_home()` caches the `CUDA_HOME` path at import time. If the environment variable changes after import, PyTorch still uses the old value. Our `setup.py` now force-refreshes `torch.utils.cpp_extension.CUDA_HOME` from the current environment before building.

2. **Missing `cuda_cmake_macros.h`** — The PyTorch 2.7 pip wheels for Windows do not include this header file, but the build process expects it. The `build_one_wheel.sh` script creates a stub file as a workaround.

3. **Missing `.lib` files** — The PyTorch pip wheels for Windows do not ship `c10_cuda.lib` or `torch_cuda.lib`, which are needed for linking. The build script generates these import libraries from the corresponding DLLs using `dumpbin` and `lib.exe`.

4. **CPU-only torch from cu121 index** — The cu121 PyTorch package index does not have CUDA wheels for Windows (only CPU-only builds). To avoid accidentally building against a CPU-only PyTorch, all Windows builds use the cu124 or cu126 indexes instead.

5. **PyTorch `Parallel.h` incompatibility** — A `static constexpr` usage in PyTorch's `Parallel.h` header is incompatible with MSVC on `windows-2022` CI runners. The build script patches this header before compiling.

6. **`.lib` files in `extra_objects`** — When reusing precompiled objects on Windows, the `precompiled/` directory may contain `.lib` files alongside `.obj` files. `CUDAExtension` does not accept `.lib` files as extra objects, so `setup.py` filters them out.

7. **`DISTUTILS_USE_SDK` not set** — When using `ilammy/msvc-dev-cmd@v1` to activate the MSVC environment, PyTorch's `cpp_extension.py` requires `DISTUTILS_USE_SDK=1` to be set. Without it, the build fails with a `UserWarning` about multiple VC environment activations. The workflow now exports this variable after the MSVC setup step.

8. **Git `link.exe` shadows MSVC `link.exe`** — The `windows-2022` runner has Git's `link.exe` at `C:\Program Files\Git\usr\bin\link.exe`, which shadows MSVC's linker on the PATH. This causes the link step to fail with exit code 1. The workflow renames Git's `link.exe` to `link_git.exe` before building.

9. **CUDA 12.8.0 installer incompatible with runner image** — The CUDA 12.8.0 installer (`cuda_12.8.0_570.86_windows.exe`) fails on the `windows-2022` GitHub runner with "This version of %1 is not compatible with the version of Windows you're running". Updated to CUDA 12.8.1 (`cuda_12.8.1_572.61_windows.exe`) which works correctly.

## Limitations and future work

- **No GPU CI tests**: The GitHub Actions CI runners do not have GPUs. The smoke tests verify binary compatibility (symbols, imports, ABI) but cannot test actual kernel correctness. To validate GPU functionality, use `scripts/test_wheels_local.py` on a machine with a CUDA-capable GPU.
- **Python 3.14**: Not included because PyTorch does not yet ship cp314 wheels. Can be added once PyTorch adds support.
- **PyTorch 2.5/2.6 backward compatibility**: Not supported. C++ ABI symbols change between PyTorch minor releases, and these symbols are used on every C++ function call via the `CHECK_INPUT` and `DEVICE_GUARD` macros. This has not been GPU-tested on any OS. Supporting older PyTorch would require building separate wheels per PyTorch version, which would triple the wheel count.
- **Windows cu118**: Not currently built. Could be added in the future since PyTorch 2.7 does ship cu118 wheels for Windows.
- **macOS**: Not applicable — macOS does not support CUDA.
