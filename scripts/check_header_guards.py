#!/usr/bin/env python3

"""Checks for missing or incorrect header guards.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

from pathlib import Path
import re
import sys
from typing import List, Optional

# Replace with RE
def get_line_with_prefix(lines: List[str], prefix: str) -> Optional[str]:
    for line in lines:
        if line.startswith(prefix):
            return line
    return None


def check_path(path: Path) -> bool:
    if path.suffix != ".h":
        print(f"Not a header: {path}", file=sys.stderr)
        return True
    with path.open() as f:
        lines = f.readlines()

    guard = str(path).upper().replace("/", "_").replace(".", "_") + "_"
    ifndef = get_line_with_prefix(lines, "#ifndef")
    define = get_line_with_prefix(lines, "#define")
    endif = get_line_with_prefix(reversed(lines), "#endif")
    if None in (ifndef, define, endif):
        print(
            f"Missing header guard in {path}: #ifndef {guard}",
            file=sys.stderr,
        )
        return True
    return False


def main() -> None:
    has_errors = False
    for arg in sys.argv[1:]:
        if check_path(Path(arg)):
            has_errors = True
    if has_errors:
        exit(1)


if __name__ == "__main__":
    main()
