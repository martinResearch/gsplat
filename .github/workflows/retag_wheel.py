#!/usr/bin/env python3
"""Retag a wheel to be installable on the specified Python versions.

Usage:
    python retag_wheel.py --tags cp310.cp311.cp312 <wheel> [<wheel> ...]
    python retag_wheel.py --tags cp313 <wheel> [<wheel> ...]

The original wheel file is replaced in-place with the retagged version.
If a single tag is given that matches the build tag, the wheel is unchanged.
"""

import argparse
import base64
import glob
import hashlib
import os
import re
import shutil
import sys
import zipfile


def retag_wheel(whl_path: str, python_tags: list[str]) -> str:
    """Retag a single wheel. Returns the new filename."""
    whl_dir = os.path.dirname(os.path.abspath(whl_path))
    whl_name = os.path.basename(whl_path)

    # Detect the build Python tag from the filename (e.g. "cp310" or "cp313")
    m = re.search(r"-(cp3\d+)-(cp3\d+)-", whl_name)
    if not m:
        print(f"WARNING: Could not detect Python tag in {whl_name}")
        return whl_name
    build_tag = m.group(1)  # e.g. "cp310" or "cp313"

    # If the only target tag matches the build tag, nothing to do
    if python_tags == [build_tag]:
        print(f"SKIP: {whl_name} already tagged {build_tag}")
        return whl_name

    tmp = os.path.join(whl_dir, "_retag_tmp")

    if os.path.exists(tmp):
        shutil.rmtree(tmp)

    # Extract
    with zipfile.ZipFile(whl_path, "r") as z:
        z.extractall(tmp)

    # Find the WHEEL metadata
    wheel_files = glob.glob(os.path.join(tmp, "*.dist-info", "WHEEL"))
    assert len(wheel_files) == 1, f"Expected 1 WHEEL file, found {len(wheel_files)}"
    wheel_file = wheel_files[0]

    with open(wheel_file, "r") as f:
        content = f.read()

    # Replace the build tag line with one line per target Python version.
    # Input:  Tag: cp310-cp310-<platform>
    # Output: Tag: cp310-cp310-<platform>\nTag: cp311-cp311-<platform>\n...
    def expand_tag(m):
        platform = m.group(1)
        return "\n".join(f"Tag: {t}-{t}-{platform}" for t in python_tags)

    content = re.sub(
        rf"Tag: {build_tag}-{build_tag}-(\S+)", expand_tag, content
    )

    with open(wheel_file, "w") as f:
        f.write(content)

    # Rebuild RECORD (hash manifest)
    dist_info = os.path.dirname(wheel_file)
    record_path = os.path.join(dist_info, "RECORD")
    records = []
    for root, _dirs, files in os.walk(tmp):
        for fname in files:
            fpath = os.path.join(root, fname)
            # ZIP spec and PEP 427 require forward slashes in archive paths.
            # os.path.relpath returns backslashes on Windows, so normalise.
            arcname = os.path.relpath(fpath, tmp).replace(os.sep, "/")
            if arcname.endswith("RECORD"):
                continue
            with open(fpath, "rb") as bf:
                data = bf.read()
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .rstrip(b"=")
                .decode()
            )
            records.append(f"{arcname},sha256={digest},{len(data)}")
    records.append(
        os.path.relpath(record_path, tmp).replace(os.sep, "/") + ",,"
    )
    with open(record_path, "w") as f:
        f.write("\n".join(records) + "\n")

    # Build new filename with compressed Python tag
    compressed = ".".join(python_tags)
    new_name = whl_name.replace(
        f"-{build_tag}-{build_tag}-", f"-{compressed}-{compressed}-"
    )
    if new_name == whl_name:
        print(f"WARNING: Could not retag {whl_name}")
        shutil.rmtree(tmp)
        return whl_name

    new_path = os.path.join(whl_dir, new_name)

    # Remove original and write retagged wheel
    os.remove(whl_path)
    with zipfile.ZipFile(new_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(tmp):
            for fname in files:
                fpath = os.path.join(root, fname)
                # Use writestr() with forward-slash arcnames.  ZipFile.write()
                # calls ZipInfo.from_file() which applies os.path.normpath(),
                # converting forward slashes back to backslashes on Windows.
                arcname = os.path.relpath(fpath, tmp).replace(os.sep, "/")
                with open(fpath, "rb") as bf:
                    z.writestr(arcname, bf.read())

    shutil.rmtree(tmp)
    return new_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retag wheels for multiple Python versions."
    )
    parser.add_argument(
        "--tags",
        required=True,
        help="Dot-separated Python tags, e.g. cp310.cp311.cp312 or cp313",
    )
    parser.add_argument("wheels", nargs="+", help="Wheel files to retag")
    args = parser.parse_args()

    tags = args.tags.split(".")
    for path in args.wheels:
        old = os.path.basename(path)
        new = retag_wheel(path, tags)
        print(f"Retagged: {old} -> {new}")
