#!/usr/bin/env python3

"""Check that a release tar contains the same files as a prefix root."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

from pathlib import Path
import os
import re
import tarfile
import unittest


class ToolchainTarTest(unittest.TestCase):
    def test_tar(self) -> None:
        install_data_manifest = Path(os.environ["INSTALL_DATA_MANIFEST"])
        tar_file = Path(os.environ["TAR_FILE"])

        # Gather install data files.
        with open(install_data_manifest) as manifest:
            # Remove everything up to and including `prefix_root`.
            install_files = set(
                [
                    re.sub("^.*/prefix_root/", "", entry.strip())
                    for entry in manifest.readlines()
                ]
            )
        self.assertTrue(install_files, f"`{install_data_manifest}` is empty.")

        # Gather tar files.
        with tarfile.open(tar_file) as tar:
            # Remove the first path component.
            tar_files = set(
                [
                    str(Path(*Path(tarinfo.name).parts[1:]))
                    for tarinfo in tar
                    if not tarinfo.isdir()
                ]
            )
        self.assertTrue(tar_files, f"`{tar_file}` is empty.")

        self.assertSetEqual(install_files, tar_files)


if __name__ == "__main__":
    unittest.main()
