#!/usr/bin/env python3

"""Integration tests for the Carbon toolchain installation."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import platform
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from bazel_tools.tools.python.runfiles import runfiles


class InstallTest(unittest.TestCase):
    def setUp(self) -> None:
        # The install root is adjacent to the test script
        self.install_root = Path(sys.argv[0]).parent
        self.tmpdir = Path(os.environ["TEST_TMPDIR"])
        self.test_o_file = self.tmpdir / "test.o"
        self.test_o_file.touch()
        self.runfiles = runfiles.Create()
        self.prebuilt_runtimes = self.runfiles.Rlocation(
            "carbon/toolchain/install/carbon_stage1_runtimes_build"
        )

    def get_link_cmd(self, clang: Path) -> list[str | Path]:
        return [
            clang,
            # Verbose printing to help with debugging.
            "-v",
            # Pass a parameter to the underlying Carbon busybox using `-Xcarbon`
            # to switch it to use the prebuilt runtimes rather than building
            # runtimes on demand.
            f"-Xcarbon=--prebuilt-runtimes={self.prebuilt_runtimes}",
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
        bin = self.install_root / "llvm/bin/clang"
        # Most errors are caught by ensuring the command succeeds.
        run = subprocess.run(
            self.get_link_cmd(bin), check=True, capture_output=True, text=True
        )

        # Also ensure that it correctly didn't imply a C++ link.
        self.assertNotRegex(run.stderr, r'"-lc\+\+"')
        self.assertNotRegex(run.stderr, r'"-lstdc\+\+"')

    # Note that we can't test `clang` vs. `clang++` portably. See the comment on
    # `test_clang` for details.
    def test_clangplusplus(self) -> None:
        bin = self.install_root / "llvm/bin/clang++"
        run = subprocess.run(
            self.get_link_cmd(bin), check=True, capture_output=True, text=True
        )

        # Ensure that this binary _does_ imply a C++ link. Also ensure it uses
        # `libc++`, as we default our Clang to use that on all platforms.
        self.assertRegex(run.stderr, r'"-lc\+\+"')

    def test_clang_cl(self) -> None:
        bin = self.install_root / "llvm/bin/clang-cl"
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
        bin = self.install_root / "llvm/bin/clang-cpp"
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

    def run_carbon_test(
        self, name: str, source: str, use_prebuilt: bool, expected_output: str
    ) -> None:
        src_file = self.tmpdir / f"{name}.carbon"
        src_file.write_text(textwrap.dedent(source).lstrip())

        output_bin = self.tmpdir / name

        carbon = self.runfiles.Rlocation(
            "carbon/toolchain/install/carbon-busybox"
        )

        try:
            obj_file = self.tmpdir / f"{name}.o"
            subprocess.run(
                [carbon, "compile", f"--output={obj_file}", src_file],
                check=True,
                capture_output=True,
                text=True,
            )

            link_cmd = [carbon]
            if use_prebuilt:
                link_cmd.append(f"--prebuilt-runtimes={self.prebuilt_runtimes}")
            link_cmd.extend(["link", f"--output={output_bin}", obj_file])
            subprocess.run(link_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as err:
            self.fail(f"Subprocess failed: {err.stderr}")

        run = subprocess.run(
            [output_bin], check=True, capture_output=True, text=True
        )
        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.stdout.strip(), expected_output)

    def run_cpp_test(
        self, name: str, source: str, use_prebuilt: bool, expected_output: str
    ) -> None:
        src_file = self.tmpdir / f"{name}.cpp"
        src_file.write_text(textwrap.dedent(source).lstrip())

        output_bin = self.tmpdir / name

        clang = self.install_root / "llvm/bin/clang++"

        try:
            cmd = [clang, f"-o{output_bin}", src_file]
            if use_prebuilt:
                cmd.append(
                    f"-Xcarbon=--prebuilt-runtimes={self.prebuilt_runtimes}"
                )
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as err:
            self.fail(f"Subprocess failed: {err.stderr}")

        run = subprocess.run(
            [output_bin], check=True, capture_output=True, text=True
        )
        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.stdout.strip(), expected_output)

    def test_carbon_end_to_end(self) -> None:
        source = r"""
        import Cpp library "<iostream>";
        fn Run() -> i32 {
          Cpp.std.cout << "Hello from Carbon\n";
          return 0;
        }
        """
        for use_prebuilt in [True, False]:
            with self.subTest(use_prebuilt=use_prebuilt):
                self.run_carbon_test(
                    "simple_carbon", source, use_prebuilt, "Hello from Carbon"
                )

    def test_cpp_end_to_end(self) -> None:
        source = r"""
        #include <iostream>
        int main() {
          std::cout << "Hello from C++" << std::endl;
          return 0;
        }
        """
        for use_prebuilt in [True, False]:
            with self.subTest(use_prebuilt=use_prebuilt):
                self.run_cpp_test(
                    "simple_cpp", source, use_prebuilt, "Hello from C++"
                )


if __name__ == "__main__":
    unittest.main()
