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
from typing import List, Optional

_CPP_REFACTORING = "./cpp_refactoring/cpp_refactoring"
_H_EXTS = {".h", ".hpp"}
_CPP_EXTS = {".c", ".cc", ".cpp", ".cxx"}


class _Workflow:
    _parsed_args: argparse.Namespace
    _data_dir: str
    _cpp_files: Optional[List[str]]

    def __init__(self) -> None:
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

    def run(self) -> None:
        """Runs the migration workflow."""
        try:
            self._gather_files()
            self._clang_tidy()
            self._cpp_refactoring()
            self._rename_files()
            self._print_header("Done!")
        except subprocess.CalledProcessError as e:
            # Discard the stack for subprocess errors.
            sys.exit(str(e))

    def _data_file(self, relative_path: str) -> str:
        """Returns the path to a data file."""
        return os.path.join(self._data_dir, relative_path)

    @staticmethod
    def _print_header(header: str) -> None:
        print("*" * 79)
        print("* %-75s *" % header)
        print("*" * 79)

    def _gather_files(self) -> None:
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

    def _clang_tidy(self) -> None:
        """Runs clang-tidy to fix C++ files in a directory."""
        self._print_header("Running clang-tidy...")
        with open(self._data_file("clang_tidy.yaml")) as f:
            config = f.read()
        subprocess.run(
            ["run-clang-tidy.py", "-fix", "-config", config],
            check=True,
        )

    def _cpp_refactoring(self) -> None:
        """Runs cpp_refactoring to migrate C++ files towards Carbon syntax."""
        self._print_header("Running cpp_refactoring...")
        cpp_refactoring = self._data_file(_CPP_REFACTORING)
        assert self._cpp_files is not None
        subprocess.run([cpp_refactoring] + self._cpp_files, check=True)

    def _rename_files(self) -> None:
        """Renames C++ files to the destination Carbon filenames."""
        api_renames = 0
        impl_renames = 0
        assert self._cpp_files is not None
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
