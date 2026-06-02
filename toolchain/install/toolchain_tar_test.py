#!/usr/bin/env python3

"""Check that a release tar contains the same files as a prefix root."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import re
import tarfile
import unittest
from pathlib import Path


class ToolchainTarTest(unittest.TestCase):
    def test_tar(self) -> None:
        install_data_manifest = Path(os.environ["INSTALL_DATA_MANIFEST"])
        tar_file = Path(os.environ["TAR_FILE"])

        # Gather install data files.
        with open(install_data_manifest) as manifest:
            # Remove everything up to and including the package path
            # `toolchain/install`.
            install_files = set(
                [
                    re.sub("^.*/toolchain/install/", "", entry.strip())
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

        # Check that the `carbon` symlink is in the tar file.
        self.assertIn("bin/carbon", tar_files)
        tar_files.remove("bin/carbon")

        # Remove the `lib/carbon` prefix which should be on every other file.
        tar_files = set(
            [re.sub("^lib/carbon/", "", entry.strip()) for entry in tar_files]
        )

        # The install files and the tar files should now be identical.
        self.assertSetEqual(install_files, tar_files)


if __name__ == "__main__":
    unittest.main()
