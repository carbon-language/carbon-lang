#!/usr/bin/env python3

"""Runs clang-tidy over all Carbon files.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import subprocess

from pathlib import Path


def main() -> None:
    # Set the repo root as the working directory.
    os.chdir(Path(__file__).parent.parent)
    # Ensure create_compdb has been run.
    subprocess.check_call(["./scripts/create_compdb.py"])
    # Run clang-tidy from clang-tools-extra.
    exit(
        subprocess.call(
            [
                "./bazel-execroot/external/llvm-project/clang-tools-extra/"
                "clang-tidy/tool/run-clang-tidy.py",
                "^(?!.*/(bazel-|third_party)).*$",
            ]
        )
    )


if __name__ == "__main__":
    main()
