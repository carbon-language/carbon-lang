#!/usr/bin/env python3

"""Checks for missing or incorrect header guards."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

from pathlib import Path
import re
import sys
from typing import Iterable, List, NamedTuple, Optional


class Guard(NamedTuple):
    """A guard line in a file."""

    line: int
    guard: str


def find_guard(
    lines: List[str], pattern: str, from_end: bool
) -> Optional[Guard]:
    """Searches the lines for something matching the pattern."""
    lines_range: Iterable[str] = lines
    if from_end:
        lines_range = reversed(lines)
    for index, line in enumerate(lines_range):
        m = re.match(pattern, line)
        if m:
            if from_end:
                index = len(lines) - index - 1
            return Guard(index, m[1])
    return None


def maybe_replace(
    lines: List[str], old_guard: Guard, guard_prefix: str, guard: str
) -> None:
    """Replaces a header guard in the file if needed."""
    if guard != old_guard.guard:
        line = lines[old_guard.line].rstrip("\n")
        print(f"- Replacing line {old_guard.line}: `{line}`", file=sys.stderr)
        lines[old_guard.line] = f"{guard_prefix} {guard}\n"


def check_path(path: Path) -> bool:
    """Checks the path for header guard issues."""
    if path.suffix != ".h":
        print(f"Not a header: {path}", file=sys.stderr)
        return True

    with path.open() as f:
        lines = f.readlines()

    guard_path = str(path).upper().replace("/", "_").replace(".", "_")
    guard = f"CARBON_{guard_path}_"
    ifndef = find_guard(lines, "#ifndef ([A-Z0-9_]+_H_)", False)
    define = find_guard(lines, "#define ([A-Z0-9_]+_H_)", False)
    endif = find_guard(lines, "#endif(?:  // ([A-Z0-9_]+_H_))?", True)
    if ifndef is None or define is None or endif is None:
        print(f"Incomplete header guard in {path}:", file=sys.stderr)
        if ifndef is None:
            print(f"- Missing `#ifndef {guard}`", file=sys.stderr)
        if define is None:
            print(f"- Missing `#define {guard}`", file=sys.stderr)
        if endif is None:
            print(f"- Missing `#endif  // {guard}`", file=sys.stderr)
        return True

    if ifndef.line + 1 != define.line:
        print(
            f"Non-consecutive header guard in {path}: "
            f"#ifndef on line {ifndef.line + 1}, "
            f"#define on line {define.line + 1}.",
            file=sys.stderr,
        )
        return True

    if endif.line != len(lines) - 1:
        print(
            f"Misordered header guard in {path}: #endif on line {endif.line}, "
            f"should be on last line ({len(lines) - 1}).",
            file=sys.stderr,
        )
        return True

    if guard != ifndef.guard or guard != define.guard or guard != endif.guard:
        print(f"Fixing header guard in {path} to {guard}:", file=sys.stderr)
        maybe_replace(lines, ifndef, "#ifndef", guard)
        maybe_replace(lines, define, "#define", guard)
        maybe_replace(lines, endif, "#endif  //", guard)
        with path.open("w") as f:
            f.writelines(lines)
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
