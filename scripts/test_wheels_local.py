#!/usr/bin/env python3
"""Test precompiled gsplat wheels locally across Python × PyTorch × CUDA combos.

Requires:
  - uv (https://docs.astral.sh/uv/getting-started/installation/)
  - An NVIDIA GPU with a recent driver

Usage:
  # Run the full matrix (all combos) from local wheel files:
  python scripts/test_wheels_local.py --wheel-dir dist/

  # Test a specific combo:
  python scripts/test_wheels_local.py --wheel-dir dist/ --python 3.12 --torch 2.6.0 --cuda cu121

  # Test wheels from a GitHub Pages index (auto-selects cu11/cu12 per test case):
  python scripts/test_wheels_local.py --index-base-url https://martinresearch.github.io/gsplat/whl

  # Test a single CUDA variant from a specific index URL:
  python scripts/test_wheels_local.py --index-url https://martinresearch.github.io/gsplat/whl/pt23to26.cu12

  # Quick smoke test only (no GPU tests):
  python scripts/test_wheels_local.py --wheel-dir dist/ --no-gpu

  # Run the full pytest suite from the repo:
  python scripts/test_wheels_local.py --wheel-dir dist/ --full-tests
"""

import argparse
import itertools
import os
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

# PyTorch versions and their supported CUDA variants + min Python
TORCH_CUDA_MATRIX = {
    "2.3.0": {"cuda": ["cu118", "cu121"], "max_python": "3.12"},
    "2.4.0": {"cuda": ["cu118", "cu121"], "max_python": "3.12"},
    "2.5.0": {"cuda": ["cu118", "cu121"], "max_python": "3.13"},
    "2.6.0": {"cuda": ["cu118", "cu121"], "max_python": "3.13"},
}

# Mapping from cu* to wheel index suffix
CUDA_TO_WHEEL_TAG = {"cu118": "cu11", "cu121": "cu12"}

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
                # Skip Windows cu118 (PyTorch doesn't publish those for 2.4+)
                if sys.platform == "win32" and cuda == "cu118" and torch_ver >= "2.4.0":
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
            # Install from local wheel files
            wheel_tag = CUDA_TO_WHEEL_TAG[case.cuda]
            # Find matching wheel: look for cu11 or cu12 in the filename
            wheel_files = list(Path(wheel_dir).glob(f"gsplat-*{wheel_tag}*.whl"))
            if not wheel_files:
                # Try with the full cuda tag
                wheel_files = list(Path(wheel_dir).glob(f"gsplat-*{case.cuda}*.whl"))
            if not wheel_files:
                print(f"  SKIP: No wheel found for {wheel_tag} in {wheel_dir}")
                return True
            wheel_file = str(wheel_files[0])
            result = run(
                [uv, "pip", "install", "--python", venv_python, wheel_file, "-q"],
                check=False,
            )
        elif index_url:
            resolved_url = index_url
            result = run(
                [uv, "pip", "install", "--python", venv_python,
                 "gsplat", "--index-url", resolved_url, "-q"],
                check=False,
            )
        else:
            print("  ERROR: need --wheel-dir, --index-url, or --index-base-url")
            return False

        if result.returncode != 0:
            print(f"  FAIL: Could not install gsplat wheel")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[:5]:
                    print(f"    {line}")
            return False

        # 4. Install test dependencies
        print(f"\n[4/5] Installing test dependencies...")
        deps = ["numpy", "jaxtyping", "typing_extensions"]
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
    source.add_argument("--wheel-dir", help="Directory containing wheel files")
    source.add_argument("--index-url", help="PEP 503 index URL for a single CUDA variant (e.g. .../pt23to26.cu12)")
    source.add_argument("--index-base-url",
                        help="Base URL of wheel index; CUDA variant is appended automatically per test case "
                             "(e.g. https://martinresearch.github.io/gsplat/whl)")

    parser.add_argument("--python", help="Specific Python version (e.g., 3.12)")
    parser.add_argument("--torch", help="Specific PyTorch version (e.g., 2.6.0)")
    parser.add_argument("--cuda", help="Specific CUDA variant (cu118 or cu121)")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU tests (smoke test only)")
    parser.add_argument("--full-tests", action="store_true",
                        help="Also run pytest tests/test_basic.py (requires GPU)")

    args = parser.parse_args()

    uv = find_uv()
    repo_root = str(Path(__file__).resolve().parent.parent)
    cases = build_matrix(args)

    if not cases:
        print("No test cases match the given filters.")
        sys.exit(1)

    print(f"Running {len(cases)} test case(s):")
    for c in cases:
        print(f"  - {c.label}")

    results: dict[str, str] = {}
    for case in cases:
        # Resolve index URL: --index-base-url constructs per-CUDA-variant URLs
        # Python 3.13+ wheels are in pt26 (Group B), all others in pt23to26 (Group A)
        if args.index_base_url:
            wheel_tag = CUDA_TO_WHEEL_TAG[case.cuda]
            pt_range = "pt26" if case.python >= "3.13" else "pt23to26"
            resolved_index_url = f"{args.index_base_url.rstrip('/')}/{pt_range}.{wheel_tag}"
        else:
            resolved_index_url = args.index_url

        ok = run_test_case(
            uv=uv,
            case=case,
            wheel_dir=args.wheel_dir,
            index_url=resolved_index_url,
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

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
