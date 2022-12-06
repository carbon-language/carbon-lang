#!/usr/bin/env python3

"""Updates the CHECK: lines in lit tests based on the AUTOUPDATE line."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import sys
from pathlib import Path


def main() -> None:
    # Calls the main script using execv in order to avoid Python import
    # behaviors.
    this_py = Path(__file__).resolve()
    actual_py = this_py.parent.parent.parent.joinpath(
        "bazel", "testing", "lit_autoupdate_base.py"
    )
    args = [
        sys.argv[0],
        # Flags to configure for explorer testing.
        "--tool=carbon",
        "--autoupdate_arg=dump",
        "--autoupdate_arg=semantics-ir",
        r"--line_number_pattern=(?<=\.carbon:)(\d+)(?=(?:\D|$))",
        "--lit_run=%{carbon-run-semantics}",
        "--testdata=toolchain/semantics/testdata",
    ] + sys.argv[1:]
    os.execv(actual_py, args)


if __name__ == "__main__":
    main()
