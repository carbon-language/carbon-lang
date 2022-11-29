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
        # Flags to configure for lexer testing.
        "--tool=carbon",
        "--autoupdate_arg=dump",
        "--autoupdate_arg=tokens",
        # Ignore the resulting column of EndOfFile because it's typically the
        # end of the CHECK comment.
        "--extra_check_replacement",
        ".*'EndOfFile'",
        r"column: (?:\d+)",
        "column: {{[0-9]+}}",
        # Ignore spaces that are used to columnize lines.
        "--line_number_format={{ *}}[[@LINE%(delta)s]]",
        r"--line_number_pattern=(?<= line: )( *\d+)(?=,)",
        "--lit_run=%{carbon-run-tokens}",
        "--testdata=toolchain/lexer/testdata",
    ] + sys.argv[1:]
    os.execv(actual_py, args)


if __name__ == "__main__":
    main()
