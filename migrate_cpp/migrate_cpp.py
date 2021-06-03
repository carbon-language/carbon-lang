"""Migrates C++ code to Carbon."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import glob
import os
import subprocess
import sys

_CLANG_TIDY = "../external/bootstrap_clang_toolchain/bin/clang-tidy"
_CPP_REFACTORING = "./cpp_refactoring/cpp_refactoring"
_H_EXTS = {".h", ".hpp"}
_CPP_EXTS = {".c", ".cc", ".cpp", ".cxx"}


class _Workflow(object):
    def __init__(self):
        """Parses command-line arguments and flags."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "dir",
            type=str,
            help="A directory containing C++ files to migrate to Carbon.",
        )
        parsed_args = parser.parse_args()
        self._parsed_args = parsed_args

        self._data_dir = os.path.dirname(sys.argv[0])

        # Validate arguments.
        if not os.path.isdir(parsed_args.dir):
            sys.exit("%r must point to a directory." % parsed_args.dir)

    def run(self):
        """Runs the migration workflow."""
        self._gather_files()
        self._clang_tidy()
        self._cpp_refactoring()
        self._rename_files()
        self._print_header("Done!")

    def _data_file(self, relative_path):
        """Returns the path to a data file."""
        return os.path.join(self._data_dir, relative_path)

    @staticmethod
    def _print_header(header):
        print("*" * 79)
        print("* %-75s *" % header)
        print("*" * 79)

    def _gather_files(self):
        """Returns the list of C++ files to convert."""
        self._print_header("Gathering C++ files...")
        all_files = glob.glob(
            os.path.join(self._parsed_args.dir, "**/*.*"), recursive=True
        )
        exts = _CPP_EXTS.union(_H_EXTS)
        cpp_files = [f for f in all_files if os.path.splitext(f)[1] in exts]
        if not cpp_files:
            sys.exit(
                "%r doesn't contain any C++ files to convert."
                % self._parsed_args.dir
            )
        self._cpp_files = sorted(cpp_files)
        print("%d files found." % len(self._cpp_files))

    def _clang_tidy(self):
        """Runs clang-tidy to fix C++ files in a directory."""
        self._print_header("Running clang-tidy...")
        clang_tidy = self._data_file(_CLANG_TIDY)
        with open(self._data_file("clang_tidy.yaml")) as f:
            config = f.read()
        subprocess.run(
            [clang_tidy, "--fix", "--config", config] + self._cpp_files
        )

    def _cpp_refactoring(self):
        """Runs cpp_refactoring to migrate C++ files towards Carbon syntax."""
        self._print_header("Running cpp_refactoring...")
        cpp_refactoring = self._data_file(_CPP_REFACTORING)
        subprocess.run([cpp_refactoring] + self._cpp_files)

    def _rename_files(self):
        """Renames C++ files to the destination Carbon filenames."""
        api_renames = 0
        impl_renames = 0
        for f in self._cpp_files:
            parts = os.path.splitext(f)
            if parts[1] in _H_EXTS:
                os.rename(f, parts[0] + ".carbon")
                api_renames += 1
            else:
                os.rename(f, parts[0] + ".impl.carbon")
                impl_renames += 1
        print(
            "Renaming resulted in %d API files and %d impl files."
            % (api_renames, impl_renames)
        )


if __name__ == "__main__":
    _Workflow().run()
