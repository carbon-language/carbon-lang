#!/usr/bin/env python3

"""Checks various LLVM tool symlinks behave as expected."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

from pathlib import Path
import subprocess
import os
import platform
import sys
import unittest


class LLVMSymlinksTest(unittest.TestCase):
    def setUp(self) -> None:
        # The install root is adjacent to the test script
        self.install_root = Path(sys.argv[0]).parent / "prefix_root"
        self.tmpdir = Path(os.environ["TEST_TMPDIR"])
        self.test_o_file = self.tmpdir / "test.o"
        self.test_o_file.touch()

    def get_link_cmd(self, clang: Path) -> list[str | Path]:
        return [
            clang,
            # Verbose printing to help with debugging.
            "-v",
            # Print out the link command rather than running it.
            "-###",
            # Give the link command an output.
            "-o",
            self.tmpdir / "test",
            # A test input file. This won't be read though.
            self.test_o_file,
        ]

    def unsupported(self, stderr: str) -> None:
        self.fail(f"Unsupported platform '{platform.uname()}':\n{stderr}")

    # Note that we can't test `clang` vs. `clang++` portably. The only commands
    # with useful differences are _link_ commands, and those need to build
    # runtime libraries on demand, which requires the host to be able to compile
    # and link for the target. Instead, we test linking with the default target
    # (the host), as that is the one that should reliably work if we're
    # developing Carbon, and encode all the different platform results in the
    # test expectations.
    def test_clang(self) -> None:
        bin = self.install_root / "lib/carbon/llvm/bin/clang"
        # Most errors are caught by ensuring the command succeeds.
        run = subprocess.run(
            self.get_link_cmd(bin), check=True, capture_output=True, text=True
        )

        # Also ensure that it correctly didn't imply a C++ link.
        if platform.system() == "Linux":
            self.assertNotRegex(run.stderr, r'"-lstdc\+\+"')
        elif platform.system() == "Darwin":
            self.assertNotRegex(run.stderr, r'"-lc\+\+"')
        else:
            self.unsupported(run.stderr)

    # Note that we can't test `clang` vs. `clang++` portably. See the comment on
    # `test_clang` for details.
    def test_clangplusplus(self) -> None:
        bin = self.install_root / "lib/carbon/llvm/bin/clang++"
        run = subprocess.run(
            self.get_link_cmd(bin), check=True, capture_output=True, text=True
        )

        # Ensure that this binary _does_ imply a C++ link.
        if platform.system() == "Linux":
            self.assertRegex(run.stderr, r'"-lstdc\+\+"')
        elif platform.system() == "Darwin":
            self.assertRegex(run.stderr, r'"-lc\+\+"')
        else:
            self.unsupported(run.stderr)

    def test_clang_cl(self) -> None:
        bin = self.install_root / "lib/carbon/llvm/bin/clang-cl"
        run = subprocess.run(
            # Use the `cl.exe`-specific help flag to test the mode.
            [bin, "/?"],
            check=True,
            capture_output=True,
            text=True,
        )

        # This should print the help string, including `cl.exe` specifics.
        self.assertRegex(run.stdout, r"CL.EXE COMPATIBILITY OPTIONS:")

    def test_clang_cpp(self) -> None:
        # Note that this is a test of the C-preprocessor mode, not C++ mode.

        # Create a test file that we'll preprocess.
        text_file = self.tmpdir / "test.txt"
        with open(text_file, "w") as f:
            f.write("TEST\n")

        # Run the preprocessor using a CPP-specific command line reading from
        # the test file and writing to stdout. We define a macro that we'll
        # check is expanded.
        bin = self.install_root / "lib/carbon/llvm/bin/clang-cpp"
        try:
            run = subprocess.run(
                [bin, "-D", "TEST=SUCCESS", text_file, "-"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as err:
            print(err.stderr, file=sys.stderr)
            raise
        self.assertEqual(run.stderr, "")
        self.assertRegex(run.stdout, r"(^|\n)SUCCESS\n")


if __name__ == "__main__":
    unittest.main()
