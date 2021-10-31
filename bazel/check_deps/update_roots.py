#!/usr/bin/env python3

"""Update the roots of the Carbon build used for dependency checking.

The dependency checking cannot use wildcard queries, so we use them here and
then create lists of relevant roots in the build file.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Change the working directory to the repository root so that the remaining
# operations reliably operate relative to that root.
os.chdir(Path(__file__).parent.parent)
directory = Path.cwd()

# Use the `BAZEL` environment variable if present. If not, then try to use
# `bazelisk` and then `bazel`.
bazel = os.environ.get("BAZEL")
if not bazel:
    bazel = "bazelisk"
    if not shutil.which(bazel):
        bazel = "bazel"
        if not shutil.which(bazel):
            sys.exit("Unable to run Bazel")

# Use the `BUILDOZER` environment variable if present. If not, then try to use
# `buildozer`.
buildozer = os.environ.get("BUILDOZER")
if not buildozer:
    buildozer = "buildozer"
    if not shutil.which(buildozer):
        sys.exit("Unable to run Buildozer")

print("Compute non-test C++ root targets...")
non_test_cc_roots_query = subprocess.run(
    [
        bazel,
        "query",
        "--noimplicit_deps",
        "--notool_deps",
        "--output=minrank",
        (
            'let non_tests = kind("cc.* rule", attr(testonly, 0, //...))'
            ' in kind("cc.* rule", deps($non_tests))'
        ),
    ],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    universal_newlines=True,
).stdout
roots = [
    target
    for rank, target in [
        line.split() for line in non_test_cc_roots_query.splitlines()
    ]
    if int(rank) == 0
]

print("Replace non-test C++ roots in the BUILD file...")
subprocess.run(
    [
        buildozer,
        "remove data",
    ]
    + ["add data '%s'" % root for root in roots]
    + ["//bazel/check_deps:non_test_cc_rules"],
    check=True,
)
