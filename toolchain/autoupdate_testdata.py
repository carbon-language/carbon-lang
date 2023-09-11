#!/usr/bin/env python3

"""Autoupdates testdata in toolchain."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    argv = [
        "bazel",
        "run",
        "-c",
        "opt",
        "//toolchain/testing:file_test",
        "--",
        "--autoupdate",
    ]
    # Support specifying tests to update, such as:
    # ./autoupdate_testdata.py lex/**/*
    if len(sys.argv) > 1:
        repo_root = Path(__file__).parent.parent
        file_tests = []
        # Filter down to just test files.
        for f in sys.argv[1:]:
            if f.endswith(".carbon"):
                path = str(Path(f).absolute().relative_to(repo_root))
                if path.find("/testdata/"):
                    file_tests.append(path)
        if not file_tests:
            sys.exit(
                f"Args do not seem to be test files; for example, {sys.argv[1]}"
            )
        argv.append("--file_tests=" + ",".join(file_tests))
    subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
