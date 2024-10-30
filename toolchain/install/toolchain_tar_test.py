#!/usr/bin/env python3

"""Check that a release tar contains the same files as a prefix root."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
from pathlib import Path
import re
import tarfile


def expect_empty_set(filename: str, file_set: set[str]) -> bool:
    """Prints and returns false when the set has entries."""
    if file_set:
        print(f"error: files only in `{filename}`:")
        for f in file_set:
            print(f"  - ${f}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tar_file",
        type=Path,
        help="The tar file to test.",
    )
    parser.add_argument(
        "install_data_manifest",
        type=Path,
        help="The install data manifest file.",
    )
    args = parser.parse_args()

    with open(args.install_data_manifest) as manifest:
        # Remove everything up to and including `prefix_root`.
        install_files = set(
            [
                re.sub("^.*/prefix_root/", "", entry.strip())
                for entry in manifest.readlines()
            ]
        )
    assert len(install_files), f"`{args.install_data_manifest}` is empty."

    # First check that every file and directory in the tar file exists in our
    # prefix root, and build a set of those paths.
    with tarfile.open(args.tar_file) as tar:
        # Remove the first path component.
        tar_files = set(
            [
                str(Path(*Path(tarinfo.name).parts[1:]))
                for tarinfo in tar
                if not tarinfo.isdir()
            ]
        )
    assert len(tar_files), f"`{args.tar_file}` is empty."

    tar_okay = expect_empty_set(args.tar_file, tar_files - install_files)
    install_okay = expect_empty_set(
        args.install_data_manifest, install_files - tar_files
    )
    if not (tar_okay and install_okay):
        exit("error: tar and install data did not match.")


if __name__ == "__main__":
    main()
