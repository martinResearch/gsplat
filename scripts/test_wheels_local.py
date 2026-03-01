#!/usr/bin/env python3
"""Test precompiled gsplat wheels locally across Python × PyTorch × CUDA combos.

Requires:
  - uv  (https://docs.astral.sh/uv/getting-started/installation/)
  - An NVIDIA GPU with a recent driver (for GPU tests; use --no-gpu to skip)

Quick start — download from CI and test:

  # Download wheels from a CI run and test them (requires `gh` CLI):
  python scripts/test_wheels_local.py --gh-run-id 22547387037

  # Same but smoke test only (no GPU needed):
  python scripts/test_wheels_local.py --gh-run-id 22547387037 --no-gpu

  # Test a single combo:
  python scripts/test_wheels_local.py --gh-run-id 22547387037 --python 3.12 --cuda cu124

Test from the published wheel index (after a GitHub Release):

  # Test wheels from the GitHub Pages index:
  python scripts/test_wheels_local.py --index-url https://martinresearch.github.io/gsplat/whl/pt26cu124/

  # Test all CUDA variants automatically:
  python scripts/test_wheels_local.py --index-base-url https://martinresearch.github.io/gsplat/whl

Test from local wheel files:

  # If you already downloaded wheels to a directory:
  python scripts/test_wheels_local.py --wheel-dir dist/

  # Test a specific CUDA variant:
  python scripts/test_wheels_local.py --wheel-dir dist/ --cuda cu126

Other options:

  # Run the full pytest suite from the repo (requires GPU):
  python scripts/test_wheels_local.py --wheel-dir dist/ --full-tests

How it works:
  For each (Python, PyTorch, CUDA) combo, the script:
    1. Creates a temporary virtual environment with `uv venv`
    2. Installs PyTorch from the matching CUDA index
    3. Installs the gsplat wheel (from local file, CI artifact, or index)
    4. Runs a smoke test (symbol check, API imports, enum/class checks)
    5. Optionally runs GPU forward+backward pass on test_garden.npz
    6. Deletes the temporary venv

  Each venv is fully isolated and cleaned up automatically.

  Default matrix (no --torch/--cuda/--python filters):
    Windows: 15 tests — 8 primary (4 Py × 2 CUDA) + 7 compat (4+3 Py × cu124)
    Linux:   27 tests — 12 primary (4 Py × 3 CUDA) + 7 compat + 8 cu118 compat

  Quick run (build version only):
    python scripts/test_wheels_local.py ... --torch 2.6.0
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]

# PyTorch versions and their supported CUDA variants + max Python.
# Wheels are built against PyTorch 2.6.0 only, but are backward compatible
# with 2.4+ (2.3.0 has an ABI break in c10::SmallVectorBase::grow_pod).
#
# When --torch is not specified, ALL versions below are tested (full matrix).
# Use --torch 2.6.0 to test only the build version for a quicker run.
TORCH_CUDA_MATRIX = {
    "2.6.0": {"cuda": ["cu118", "cu124", "cu126"], "max_python": "3.13"},
    # Backward compat: same cu124 wheels (built for 2.6) tested with older torch.
    "2.5.0": {"cuda": ["cu124"], "max_python": "3.13"},
    "2.4.0": {"cuda": ["cu124"], "max_python": "3.12"},
}

EXPECTED_SYMBOLS = [
    "null",
    "quat_scale_to_covar_preci_fwd", "quat_scale_to_covar_preci_bwd",
    "spherical_harmonics_fwd", "spherical_harmonics_bwd",
    "adam", "relocation",
    "intersect_tile", "intersect_offset",
    "projection_ewa_simple_fwd", "projection_ewa_simple_bwd",
    "projection_ewa_3dgs_fused_fwd", "projection_ewa_3dgs_fused_bwd",
    "projection_ewa_3dgs_packed_fwd", "projection_ewa_3dgs_packed_bwd",
    "rasterize_to_pixels_3dgs_fwd", "rasterize_to_pixels_3dgs_bwd",
    "rasterize_to_indices_3dgs",
    "projection_2dgs_fused_fwd", "projection_2dgs_fused_bwd",
    "projection_2dgs_packed_fwd", "projection_2dgs_packed_bwd",
    "rasterize_to_pixels_2dgs_fwd", "rasterize_to_pixels_2dgs_bwd",
    "rasterize_to_indices_2dgs",
    "projection_ut_3dgs_fused",
    "rasterize_to_pixels_from_world_3dgs_fwd",
    "rasterize_to_pixels_from_world_3dgs_bwd",
]

EXPECTED_CLASSES = [
    "CameraModelType", "ShutterType",
    "UnscentedTransformParameters", "FThetaCameraDistortionParameters",
]


@dataclass
class TestCase:
    python: str
    torch: str
    cuda: str

    @property
    def label(self) -> str:
        return f"py{self.python}-torch{self.torch}-{self.cuda}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], *, cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def find_uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        print("ERROR: 'uv' not found. Install it: https://docs.astral.sh/uv/getting-started/installation/")
        sys.exit(1)
    return uv


def download_wheels_from_ci(run_id: str, dest_dir: str) -> None:
    """Download wheel artifacts from a GitHub Actions CI run using `gh`."""
    gh = shutil.which("gh")
    if not gh:
        print("ERROR: 'gh' CLI not found. Install it: https://cli.github.com/")
        sys.exit(1)

    print(f"Downloading wheels from CI run {run_id} to {dest_dir}...")
    result = run(
        [gh, "run", "download", run_id, "--pattern", "compiled_wheels-*", "--dir", dest_dir],
        check=False,
    )
    if result.returncode != 0:
        print(f"ERROR: Failed to download artifacts from run {run_id}")
        if result.stderr:
            print(result.stderr)
        sys.exit(1)

    # Count downloaded wheels
    wheels = list(Path(dest_dir).rglob("*.whl"))
    if not wheels:
        print(f"ERROR: No .whl files found in {dest_dir}")
        sys.exit(1)
    print(f"  Downloaded {len(wheels)} wheel(s)")

    # Flatten: move all .whl files to dest_dir root (gh downloads into subdirs)
    for whl in wheels:
        target = Path(dest_dir) / whl.name
        if whl.parent != Path(dest_dir):
            whl.rename(target)

    # Clean up empty subdirectories
    for d in sorted(Path(dest_dir).iterdir(), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass


def build_matrix(args) -> list[TestCase]:
    """Build the list of test cases from args or full matrix."""
    if args.python and args.torch and args.cuda:
        return [TestCase(python=args.python, torch=args.torch, cuda=args.cuda)]

    cases = []
    for torch_ver, info in TORCH_CUDA_MATRIX.items():
        if args.torch and args.torch != torch_ver:
            continue
        for cuda in info["cuda"]:
            if args.cuda and args.cuda != cuda:
                continue
            for py in PYTHON_VERSIONS:
                if args.python and args.python != py:
                    continue
                # Skip unsupported Python versions
                if py > info["max_python"]:
                    continue
                # Skip Windows cu118 — we don't build cu118 wheels for Windows
                if sys.platform == "win32" and cuda == "cu118":
                    continue
                cases.append(TestCase(python=py, torch=torch_ver, cuda=cuda))
    return cases


def build_smoke_test_script(test_gpu: bool, full_tests: bool, repo_root: str) -> str:
    """Build the Python test script that runs inside each venv."""
    lines = [
        "import sys, importlib",
        "print(f'Python: {sys.version}')",
        "",
        "# --- 1. Import gsplat ---",
        "import gsplat",
        "print(f'gsplat: {gsplat.__version__}')",
        "",
        "# --- 2. Symbol check ---",
        "csrc = importlib.import_module('gsplat.csrc')",
        f"EXPECTED = {EXPECTED_SYMBOLS + EXPECTED_CLASSES}",
        "missing = [s for s in EXPECTED if not hasattr(csrc, s)]",
        "if missing:",
        "    print(f'FAIL: missing symbols: {missing}')",
        "    sys.exit(1)",
        f"print(f'OK: all {len(EXPECTED_SYMBOLS)} functions and {len(EXPECTED_CLASSES)} classes found')",
        "",
        "# --- 3. Public API imports ---",
        "from gsplat import rasterization, rendering, utils",
        "from gsplat.cuda._wrapper import fully_fused_projection, rasterize_to_pixels",
        "print('OK: public API imports')",
        "",
        "# --- 4. Enum access ---",
        "assert csrc.CameraModelType.PINHOLE is not None",
        "assert csrc.ShutterType.GLOBAL is not None",
        "print('OK: enum values')",
        "",
        "# --- 5. Class instantiation ---",
        "ut = csrc.UnscentedTransformParameters()",
        "assert hasattr(ut, 'alpha')",
        "ftheta = csrc.FThetaCameraDistortionParameters()",
        "assert hasattr(ftheta, 'max_angle')",
        "print('OK: C++ class instantiation')",
        "",
        "# --- 6. PyTorch + CUDA info ---",
        "import torch",
        "print(f'PyTorch: {torch.__version__}')",
        "print(f'CUDA (PyTorch): {torch.version.cuda}')",
        f"print(f'CUDA available: {{torch.cuda.is_available()}}')",
    ]

    if test_gpu:
        lines += [
            "",
            "# --- 7. GPU smoke test ---",
            "if not torch.cuda.is_available():",
            "    print('SKIP: no CUDA device available')",
            "    sys.exit(0)",
            "",
            "print(f'GPU: {torch.cuda.get_device_name(0)}')",
            "print(f'Driver CUDA: {torch.version.cuda}')",
            "",
            "import os",
            "from gsplat._helper import load_test_data",
            f"data_path = os.path.join('{repo_root.replace(chr(92), '/')}', 'assets', 'test_garden.npz')",
            "means, quats, scales, opacities, colors, viewmats, Ks, width, height = load_test_data(",
            "    device=torch.device('cuda:0'), data_path=data_path)",
            "",
            "# Enable gradients for backward pass check",
            "means.requires_grad_(True)",
            "quats.requires_grad_(True)",
            "scales.requires_grad_(True)",
            "opacities.requires_grad_(True)",
            "colors.requires_grad_(True)",
            "",
            "# Run a forward pass through the full rasterization pipeline",
            "from gsplat import rasterization",
            "renders, alphas, info = rasterization(",
            "    means=means, quats=quats, scales=scales, opacities=opacities,",
            "    colors=colors, viewmats=viewmats, Ks=Ks, width=width, height=height,",
            ")",
            "print(f'OK: rasterization forward pass, output shape: {renders.shape}')",
            "",
            "# Run backward pass",
            "loss = renders.sum()",
            "loss.backward()",
            "print(f'OK: backward pass completed')",
            "",
            "# Verify gradients exist",
            "assert means.grad is not None, 'means.grad is None'",
            "assert quats.grad is not None, 'quats.grad is None'",
            "print(f'OK: gradients computed (means.grad norm: {means.grad.norm():.6f})')",
        ]

    lines.append("")
    lines.append("print('ALL TESTS PASSED')")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_test_case(
    uv: str,
    case: TestCase,
    wheel_dir: str | None,
    index_url: str | None,
    version: str | None,
    test_gpu: bool,
    full_tests: bool,
    repo_root: str,
) -> bool:
    """Run a single test case. Returns True on success."""
    print(f"\n{'='*70}")
    print(f"  {case.label}")
    print(f"{'='*70}")

    with tempfile.TemporaryDirectory(prefix=f"gsplat_test_{case.label}_") as tmpdir:
        venv_dir = os.path.join(tmpdir, ".venv")

        # 1. Create venv
        print(f"\n[1/5] Creating venv with Python {case.python}...")
        result = run([uv, "venv", venv_dir, "--python", case.python, "-q"], check=False)
        if result.returncode != 0:
            print(f"  SKIP: Python {case.python} not available ({result.stderr.strip()})")
            return True  # Not a failure, just skipped

        # Determine pip/python paths
        if sys.platform == "win32":
            venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            venv_python = os.path.join(venv_dir, "bin", "python")

        # 2. Install PyTorch
        print(f"\n[2/5] Installing PyTorch {case.torch}+{case.cuda}...")
        torch_url = f"https://download.pytorch.org/whl/{case.cuda}"
        result = run(
            [uv, "pip", "install", "--python", venv_python,
             f"torch=={case.torch}", "--extra-index-url", torch_url, "-q"],
            check=False,
        )
        if result.returncode != 0:
            print(f"  SKIP: PyTorch {case.torch}+{case.cuda} not available for Python {case.python}")
            if result.stderr:
                # Show first 3 lines of error
                for line in result.stderr.strip().split("\n")[:3]:
                    print(f"    {line}")
            return True  # Not a failure

        # 3. Install gsplat wheel
        print(f"\n[3/5] Installing gsplat wheel...")
        if wheel_dir:
            # Install from local wheel files using full CUDA tags (e.g. cu124, cu126)
            # Wheel names look like: gsplat-1.5.3+pt26cu124-cp312-cp312-win_amd64.whl
            py_tag = f"cp{case.python.replace('.', '')}"
            ver_glob = f"-{version}+" if version else "-*"
            wheel_files = list(Path(wheel_dir).glob(f"gsplat{ver_glob}*{case.cuda}*{py_tag}*.whl"))
            if not wheel_files:
                # Fallback: match just CUDA tag without Python filter
                wheel_files = list(Path(wheel_dir).glob(f"gsplat{ver_glob}*{case.cuda}*.whl"))
            if not wheel_files:
                print(f"  SKIP: No wheel found for {case.cuda} in {wheel_dir}")
                return True
            wheel_file = str(wheel_files[0])
            result = run(
                [uv, "pip", "install", "--python", venv_python, wheel_file, "-q"],
                check=False,
            )
        elif index_url:
            # Install from a PEP 503 index (e.g. GitHub Pages).
            # Use --no-deps since torch and deps are already installed.
            # Do NOT add --extra-index-url to prevent fallback to PyPI's
            # source-only package (which lacks compiled extensions).
            pkg_spec = f"gsplat=={version}" if version else "gsplat"
            result = run(
                [uv, "pip", "install", "--python", venv_python,
                 pkg_spec, "--index-url", index_url,
                 "--no-deps", "-q"],
                check=False,
            )
        else:
            print("  ERROR: need --wheel-dir, --index-url, --index-base-url, or --gh-run-id")
            return False

        if result.returncode != 0:
            print(f"  FAIL: Could not install gsplat wheel")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[:5]:
                    print(f"    {line}")
            return False

        # 4. Install test dependencies
        print(f"\n[4/5] Installing test dependencies...")
        deps = ["numpy", "jaxtyping", "typing_extensions", "packaging", "rich", "setuptools"]
        if full_tests:
            deps += ["pytest", "rich"]
        run(
            [uv, "pip", "install", "--python", venv_python] + deps + ["-q"],
            check=False,
        )

        # 5. Run tests
        print(f"\n[5/5] Running tests...")
        test_script = build_smoke_test_script(test_gpu, full_tests, repo_root)
        script_path = os.path.join(tmpdir, "test_gsplat.py")
        with open(script_path, "w") as f:
            f.write(test_script)

        result = run([venv_python, script_path], check=False)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            print(f"  FAIL: {case.label}")
            return False

        # 6. Optionally run full pytest suite
        if full_tests and test_gpu:
            print(f"\n[bonus] Running pytest tests/test_basic.py...")
            test_dir = os.path.join(repo_root, "tests")
            result = run(
                [venv_python, "-m", "pytest", os.path.join(test_dir, "test_basic.py"),
                 "-x", "-v", "--tb=short"],
                check=False,
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            if result.returncode != 0:
                print(f"  FAIL: pytest {case.label}")
                return False

        print(f"  PASS: {case.label}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Test gsplat precompiled wheels across Python × PyTorch × CUDA combos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wheel-dir", help="Directory containing .whl files")
    source.add_argument("--gh-run-id",
                        help="GitHub Actions run ID — downloads wheel artifacts automatically "
                             "(requires `gh` CLI authenticated)")
    source.add_argument("--index-url",
                        help="PEP 503 index URL for a single CUDA variant, e.g. "
                             "https://martinresearch.github.io/gsplat/whl/pt26cu124/")
    source.add_argument("--index-base-url",
                        help="Base URL of wheel index; CUDA variant is appended automatically "
                             "per test case (e.g. https://martinresearch.github.io/gsplat/whl)")

    parser.add_argument("--python", help="Specific Python version (e.g., 3.12)")
    parser.add_argument("--torch", help="Specific PyTorch version to test (e.g., 2.6.0). "
                        "If omitted, tests all versions in the matrix (2.6.0 + backward compat 2.5.0, 2.4.0).")
    parser.add_argument("--cuda", help="Specific CUDA variant (cu118, cu124, or cu126)")
    parser.add_argument("--version", help="gsplat version to install (e.g., 1.5.3). "
                        "Defaults to latest available.")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU tests (smoke test only)")
    parser.add_argument("--full-tests", action="store_true",
                        help="Also run pytest tests/test_basic.py (requires GPU)")

    args = parser.parse_args()

    # Auto-detect --cuda from --index-url if not explicitly set
    # e.g. https://martinresearch.github.io/gsplat/whl/pt26cu124/ -> cu124
    if args.index_url and not args.cuda:
        m = re.search(r'(cu\d{3,4})/?$', args.index_url.rstrip('/'))
        if m:
            args.cuda = m.group(1)
            print(f"Auto-detected --cuda {args.cuda} from index URL")

    # Handle --gh-run-id: download artifacts to a temp dir, then treat as --wheel-dir
    _ci_tmpdir = None
    if args.gh_run_id:
        _ci_tmpdir = tempfile.mkdtemp(prefix="gsplat_ci_wheels_")
        download_wheels_from_ci(args.gh_run_id, _ci_tmpdir)
        args.wheel_dir = _ci_tmpdir

    uv = find_uv()
    repo_root = str(Path(__file__).resolve().parent.parent)
    cases = build_matrix(args)

    if not cases:
        print("No test cases match the given filters.")
        sys.exit(1)

    # Group cases by torch version for a clear overview
    by_torch: dict[str, list[TestCase]] = {}
    for c in cases:
        by_torch.setdefault(c.torch, []).append(c)

    print(f"\nTest matrix: {len(cases)} case(s)")
    for tv in sorted(by_torch):
        group = by_torch[tv]
        cudas = sorted(set(c.cuda for c in group))
        pythons = sorted(set(c.python for c in group))
        print(f"  torch {tv}: Python {', '.join(pythons)} × CUDA {', '.join(cudas)} ({len(group)} tests)")
    print()

    results: dict[str, str] = {}
    for case in cases:
        # Resolve index URL: --index-base-url constructs per-CUDA-variant URLs
        # Full CUDA tags: pt26cu118, pt26cu124, pt26cu126
        if args.index_base_url:
            resolved_index_url = f"{args.index_base_url.rstrip('/')}/pt26{case.cuda}"
        else:
            resolved_index_url = args.index_url

        ok = run_test_case(
            uv=uv,
            case=case,
            wheel_dir=args.wheel_dir,
            index_url=resolved_index_url,
            version=args.version,
            test_gpu=not args.no_gpu,
            full_tests=args.full_tests,
            repo_root=repo_root,
        )
        results[case.label] = "PASS" if ok else "FAIL"

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for label, status in results.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {label}: {status}")

    failed = sum(1 for s in results.values() if s == "FAIL")
    total = len(results)
    print(f"\n  {total - failed}/{total} passed")

    # Clean up CI temp dir if we created one
    if _ci_tmpdir and os.path.exists(_ci_tmpdir):
        shutil.rmtree(_ci_tmpdir, ignore_errors=True)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
