# Installing `gsplat` from Pre-compiled Wheels

Pre-compiled wheels contain CUDA binaries and do **not** require a CUDA toolkit, Visual Studio, or any compiler on your machine. They are available for **Linux** and **Windows** on **Python 3.10 – 3.13**.

## Supported Versions

Each wheel is built for a specific CUDA version and tested across a range of PyTorch versions. The wheel name encodes the compatibility range:

| Wheel tag | CUDA | PyTorch | Python |
|---|---|---|---|
| `pt23to26.cu11` | 11.x | 2.3 – 2.6 | 3.10 – 3.12 |
| `pt23to26.cu12` | 12.x | 2.3 – 2.6 | 3.10 – 3.12 |
| `pt26.cu11` | 11.x | 2.6+ | 3.13 |
| `pt26.cu12` | 12.x | 2.6+ | 3.13 |

> **Why two PyTorch ranges?** Python 3.13 removed an internal CPython symbol
> (`_PyThreadState_UncheckedGet`) that pybind11 references, so 3.13 needs a
> separately compiled wheel. PyTorch 2.6 is the earliest version available
> for Python 3.13 on the PyTorch index.

> **How to choose:** match the CUDA version of your PyTorch installation.
> Run `python -c "import torch; print(torch.version.cuda)"` to check.
> - If it prints `11.*` → use `cu11`
> - If it prints `12.*` → use `cu12`

## Installation

### Step 1 – Install PyTorch

Install PyTorch first if you haven't already ([pytorch.org](https://pytorch.org/get-started/locally/)).

### Step 2 – Install gsplat

Pick the index URL that matches your CUDA version:

**CUDA 12.x** (most common):
```bash
# Python 3.10 – 3.12:
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt23to26.cu12
# Python 3.13:
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt26.cu12
```

**CUDA 11.x**:
```bash
# Python 3.10 – 3.12:
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt23to26.cu11
# Python 3.13:
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt26.cu11
```

> **Note:** You may also need to install gsplat's dependencies manually since
> `--index-url` overrides the default PyPI index:
> ```bash
> pip install ninja numpy jaxtyping "rich>=12"
> pip install gsplat --index-url https://docs.gsplat.studio/whl/pt23to26.cu12
> ```

### Pinning a specific version

If you need a specific gsplat version, you can pin it:
```bash
pip install gsplat==1.5.3+pt23to26.cu12 --index-url https://docs.gsplat.studio/whl
```

## Browsing available wheels

All published wheels are listed at [https://docs.gsplat.studio/whl/gsplat/](https://docs.gsplat.studio/whl/gsplat/).

## Troubleshooting

- **"No matching distribution found"** – Your Python or PyTorch version may not be in the supported range. Fall back to installing from source (see below) or from PyPI (`pip install gsplat`, which JIT-compiles CUDA code on first import).
- **Import errors about CUDA version mismatch** – Make sure the `cu11` / `cu12` tag matches your PyTorch's CUDA version, not necessarily the system CUDA driver version.

## Alternative: install from source

If no pre-compiled wheel matches your setup, see the [main README](../README.md#installation) for instructions on installing from source.
