#!/usr/bin/env python3

"""Check that a release tar contains the same files as a prefix root."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import sys
from pathlib import Path
import tarfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tar_file",
        type=Path,
        help="The tar file to test.",
    )
    parser.add_argument(
        "install_marker",
        type=Path,
        help="The path of the install marker in a prefix root to test against.",
    )
    args = parser.parse_args()

    # Locate the prefix root from the install marker.
    if not args.install_marker.exists():
        sys.exit("ERROR: No install marker: " + args.install_marker)
    prefix_root_path = args.install_marker.parents[2]

    # First check that every file and directory in the tar file exists in our
    # prefix root, and build a set of those paths.
    installed_paths = set()
    with tarfile.open(args.tar_file) as tar:
        for tarinfo in tar:
            relative_path = Path(*Path(tarinfo.name).parts[1:])
            installed_paths.add(relative_path)
            if not prefix_root_path.joinpath(relative_path).exists():
                sys.exit(
                    "ERROR: File `{0}` is not in prefix root: `{1}`".format(
                        tarinfo.name, prefix_root_path
                    )
                )

    # If we found an empty tar file, it's always an error.
    if len(installed_paths) == 0:
        sys.exit("ERROR: Tar file `{0}` was empty.".format(args.tar_file))

    # Now check that every file and directory in the prefix root is in that set.
    for prefix_path in prefix_root_path.glob("**/*"):
        relative_path = prefix_path.relative_to(prefix_root_path)
        if relative_path not in installed_paths:
            sys.exit(
                "ERROR: File `{0}` is not in tar file.".format(relative_path)
            )


if __name__ == "__main__":
    main()
