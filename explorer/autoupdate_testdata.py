#!/usr/bin/env python3

"""Updates the CHECK: lines in tests with an AUTOUPDATE line."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess
import sys
import glob
from pathlib import Path


def main() -> None:
    # Subprocess to the main script in order to avoid Python import behaviors.
    this_py = Path(__file__).resolve()
    autoupdate_py = this_py.parent.parent.joinpath(
        "testing", "scripts", "autoupdate_testdata_base.py"
    )
    # Get all subdirectories within "explorer/testdata" except "trace"
    subdirectories = [
        dir for dir in glob.glob("explorer/testdata/*/") if "trace" not in dir
    ]
    args = [
        str(autoupdate_py),
        # Flags to configure for explorer testing.
        "--tool=explorer",
        "--testdata=" + ",".join(subdirectories),
    ] + sys.argv[1:]
    subprocess.call(args)

    args = [
        str(autoupdate_py),
        # Flags to configure for explorer testing.
        "--tool=explorer",
        "--testdata=explorer/testdata/trace",
        "--autoupdate_arg=--trace_file=-",
        "--autoupdate_arg=-trace_all",
    ] + sys.argv[1:]
    exit(subprocess.call(args))


if __name__ == "__main__":
    main()
