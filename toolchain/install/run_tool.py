#!/usr/bin/env python3

"""Runs the tool specified in argv.

This script is essentially just a bounce-through to get an appropriate arg0.
See the TODO in run_tool.bzl.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    symbolizer = str(Path(os.environ["LLVM_SYMBOLIZER_PATH"]).absolute())
    os.environ["LLVM_SYMBOLIZER_PATH"] = symbolizer
    os.execv(sys.argv[1], sys.argv[1:])
