"""Check which DLL symbols gsplat.csrc needs that are missing from the current torch.

Usage:
    python scripts/check_dll_imports.py <path_to_csrc.pyd> <torch_lib_dir>

This uses pefile to parse PE imports from the .pyd and compare against
the torch DLL exports. Falls back to ctypes LoadLibrary diagnostics.
"""

import importlib
import os
import struct
import subprocess
import sys
import traceback


def find_vs_dumpbin():
    """Try to find dumpbin from Visual Studio."""
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if not os.path.exists(vswhere):
        return None
    try:
        result = subprocess.run(
            [vswhere, "-latest", "-property", "installationPath"],
            capture_output=True, text=True, check=True,
        )
        vs_path = result.stdout.strip()
        # Look for dumpbin in MSVC tools
        for root, dirs, files in os.walk(os.path.join(vs_path, "VC", "Tools", "MSVC")):
            for f in files:
                if f.lower() == "dumpbin.exe":
                    return os.path.join(root, f)
    except Exception:
        pass
    return None


def parse_pe_imports(pyd_path):
    """Parse PE imports using struct (no pefile dependency)."""
    imports = {}
    try:
        import pefile
        pe = pefile.PE(pyd_path)
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll_name = entry.dll.decode()
            symbols = []
            for imp in entry.imports:
                if imp.name:
                    symbols.append(imp.name.decode())
                else:
                    symbols.append(f"ordinal_{imp.ordinal}")
            imports[dll_name] = symbols
        pe.close()
    except ImportError:
        print("pefile not available, trying dumpbin...")
        dumpbin = find_vs_dumpbin()
        if dumpbin:
            result = subprocess.run(
                [dumpbin, "/IMPORTS", pyd_path],
                capture_output=True, text=True,
            )
            current_dll = None
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.endswith(".dll") or line.endswith(".DLL"):
                    current_dll = line
                    imports[current_dll] = []
                elif current_dll and line and not line.startswith("Section") and not line.startswith("Microsoft"):
                    # Import lines look like "  1A3  some_symbol_name"
                    parts = line.split(None, 1)
                    if len(parts) == 2 and parts[0].isalnum():
                        imports[current_dll].append(parts[1])
        else:
            print("Neither pefile nor dumpbin available.")
    return imports


def check_dll_exports(dll_path):
    """Get exported symbols from a DLL."""
    exports = set()
    try:
        import pefile
        pe = pefile.PE(dll_path)
        if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
            for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                if exp.name:
                    exports.add(exp.name.decode())
        pe.close()
    except ImportError:
        dumpbin = find_vs_dumpbin()
        if dumpbin:
            result = subprocess.run(
                [dumpbin, "/EXPORTS", dll_path],
                capture_output=True, text=True,
            )
            for line in result.stdout.splitlines():
                parts = line.strip().split()
                if len(parts) >= 4 and parts[0].isdigit():
                    exports.add(parts[3])
    return exports


def main():
    # First, just try the import and see the error
    print(f"Python: {sys.version}")
    print(f"CWD: {os.getcwd()}")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        print(f"Torch lib dir: {torch_lib}")
    except Exception as e:
        print(f"Failed to import torch: {e}")
        return

    try:
        csrc = importlib.import_module("gsplat.csrc")
        print("SUCCESS: gsplat.csrc loaded fine!")
        return
    except ImportError as e:
        print(f"\nImportError: {e}")
        traceback.print_exc()
        print()

    # Find the .pyd
    try:
        import gsplat
        gsplat_dir = os.path.dirname(gsplat.__file__)
        pyd_path = os.path.join(gsplat_dir, "csrc.pyd")
        print(f"PYD path: {pyd_path}")
        print(f"PYD exists: {os.path.exists(pyd_path)}")
    except Exception as e:
        print(f"Could not find gsplat path: {e}")
        return

    if not os.path.exists(pyd_path):
        print("ERROR: csrc.pyd not found")
        return

    # Try to install pefile for analysis
    try:
        import pefile
    except ImportError:
        print("Installing pefile for PE analysis...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pefile", "-q"], check=True)
        import pefile

    # Parse imports from the .pyd
    print("\n=== Parsing .pyd imports ===")
    pe = pefile.PE(pyd_path)
    torch_related = {}
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        dll_name = entry.dll.decode()
        if any(kw in dll_name.lower() for kw in ["torch", "c10", "caffe2"]):
            symbols = []
            for imp in entry.imports:
                if imp.name:
                    symbols.append(imp.name.decode())
                else:
                    symbols.append(f"ordinal_{imp.ordinal}")
            torch_related[dll_name] = symbols
            print(f"\n{dll_name}: {len(symbols)} imports")
    pe.close()

    # Check each torch DLL for missing symbols
    print("\n=== Checking for missing symbols ===")
    for dll_name, needed_symbols in torch_related.items():
        dll_path = os.path.join(torch_lib, dll_name)
        if not os.path.exists(dll_path):
            print(f"\nMISSING DLL: {dll_name} not found in {torch_lib}")
            continue

        pe_dll = pefile.PE(dll_path)
        exported = set()
        if hasattr(pe_dll, 'DIRECTORY_ENTRY_EXPORT'):
            for exp in pe_dll.DIRECTORY_ENTRY_EXPORT.symbols:
                if exp.name:
                    exported.add(exp.name.decode())
        pe_dll.close()

        missing = [s for s in needed_symbols if s not in exported]
        if missing:
            print(f"\n{dll_name}: {len(missing)} MISSING symbols:")
            for s in missing:
                print(f"  - {s}")
        else:
            print(f"\n{dll_name}: all {len(needed_symbols)} symbols present")


if __name__ == "__main__":
    main()
