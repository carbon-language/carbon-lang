#!/usr/bin/env python3

"""Autoupdates testdata in toolchain."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import subprocess

TARGETS = {
    "check": "//toolchain/check:check_file_test",
    "codegen": "//toolchain/codegen:codegen_file_test",
    "driver": "//toolchain/driver:driver_file_test",
    "lex": "//toolchain/lex:lex_file_test",
    "lower": "//toolchain/lower:lower_file_test",
    "parse": "//toolchain/parse:parse_file_test",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dirs",
        # We don't use `choices` because it seems to conflict with "*".
        nargs="*",
        default=TARGETS.keys(),
        help="Optionally restrict directories to update. Defaults to all.",
    )
    parsed_args = parser.parse_args()

    # Deduplicate and validate arguments.
    dirs = set(parsed_args.dirs)
    invalid_dirs = dirs.difference(TARGETS.keys())
    if invalid_dirs:
        exit(
            f"Invalid dirs: {', '.join(invalid_dirs)}; "
            f"allowed dirs are {', '.join(TARGETS.keys())}."
        )

    # Build the targets together if there's more than one. Otherwise, we may as
    # well build and run together.
    if len(dirs) > 1:
        subprocess.check_call(
            ["bazel", "build", "-c", "opt"] + [TARGETS[d] for d in dirs]
        )
    for d in dirs:
        subprocess.check_call(
            ["bazel", "run", "-c", "opt", TARGETS[d], "--", "--autoupdate"]
        )


if __name__ == "__main__":
    main()
