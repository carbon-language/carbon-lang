#!/usr/bin/env python3

"""Updates the CHECK: lines in tests with an AUTOUPDATE line."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Subprocess to the main script in order to avoid Python import behaviors.
    this_py = Path(__file__).resolve()
    autoupdate_py = this_py.parent.parent.joinpath(
        "testing", "scripts", "autoupdate_testdata_base.py"
    )
    args = [
        str(autoupdate_py),
        # Flags to configure for explorer testing.
        "--tool=explorer",
        "--testdata=explorer/lit_testdata",
        "--lit_run=%{explorer} | %{FileCheck-strict}",
        "--lit_run=%{explorer} --parser_debug --trace_file=- | "
        "%{FileCheck-allow-unmatched}",
    ] + sys.argv[1:]
    exit(subprocess.call(args))


if __name__ == "__main__":
    main()
