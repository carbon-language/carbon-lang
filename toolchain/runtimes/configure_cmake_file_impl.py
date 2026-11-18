#!/usr/bin/env python3
"""Script to apply a set of defines to a CMake-style configure file.

This serves as the action implementation for `configure_cmake_file.bzl`. See the
documentation in the rule of that file for more details about how to use this,
or `--help` on the script.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import re
from typing import Dict

# A set of CMake values that are considered "false".
# Based on https://cmake.org/cmake/help/latest/command/if.html
_CMAKE_FALSE_VALUES = {
    "",
    "0",
    "OFF",
    "NO",
    "N",
    "FALSE",
    "IGNORE",
    "NOTFOUND",
}

_VAR_AT_PATTERN = re.compile(r"@([^@]*)@")
_VAR_DOLLAR_PATTERN = re.compile(r"${([^}]*)}")

_DIRECTIVE_PATTERN = re.compile(
    r"^#(?P<indent>[ \t]*)cmakedefine\s+(?P<var>\w+)(?P<rest>.*)?$"
)
_DIRECTIVE_01_PATTERN = re.compile(
    r"^#(?P<indent>[ \t]*)cmakedefine01\s+(?P<var>\w+)$"
)


def _is_cmake_true(value: str) -> bool:
    """Returns true if the value is not a CMake false value.

    This is how CMake defines values as 'true' vs. 'false':
    https://cmake.org/cmake/help/latest/command/if.html
    """
    return (
        value.upper() not in _CMAKE_FALSE_VALUES
        and not value.upper().endswith("-NOTFOUND")
    )


def _substitute_variables(text: str, defines: Dict[str, str]) -> str:
    """Substitutes @VAR@ and ${VAR} style variables in a string."""

    def repl(m: re.Match[str]) -> str:
        return defines.get(str(m.group(1)), "")

    return re.sub(
        _VAR_AT_PATTERN, repl, re.sub(_VAR_DOLLAR_PATTERN, repl, text)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--defines", nargs=2, action="append", default=[])
    args = parser.parse_args()

    defines = dict(args.defines)

    with open(args.src, "r") as f:
        content = f.read()

    output_lines = []
    for line in content.splitlines():
        if m := re.match(_DIRECTIVE_PATTERN, line):
            var = m.group("var")
            if var in defines and _is_cmake_true(defines[var]):
                rest = _substitute_variables(m.group("rest"), defines)
                output_lines.append(
                    "#%sdefine %s %s" % (m.group("indent"), var, rest)
                )
            else:
                # The variable is false, so leave it undefined.
                output_lines.append("/* #undef %s */" % var)
        elif m := re.match(_DIRECTIVE_01_PATTERN, line):
            var = m.group("var")
            indent = m.group("indent")
            if var in defines and _is_cmake_true(defines[var]):
                output_lines.append("#%sdefine %s 1" % (indent, var))
            else:
                output_lines.append("#%sdefine %s 0" % (indent, var))
        else:
            output_lines.append(_substitute_variables(line, defines))

    with open(args.out, "w") as f:
        f.write("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()
