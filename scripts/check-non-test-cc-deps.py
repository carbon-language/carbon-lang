#!/usr/bin/env python3

"""Check that the dependencies of non-test C++ rules are correct.

Carbon works to ensure its user-visible libraries and binaries have a single,
simple license used for the whole project as well as for LLVM itself. However,
we frequently use third-party projects and libraries where useful in our test
code. Here, we verify that the dependencies of non-test C++ rules only include
Carbon and LLVM code.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import sys
from pathlib import Path

runfiles = Path(os.environ.get("TEST_SRCDIR"))
try:
    deps_path = runfiles / "carbon" / "non-test-cc-deps.txt"
    with deps_path.open() as deps_file:
        deps = deps_file.read().splitlines()
except FileNotFoundError:
    sys.exit("ERROR: unable to find deps file: " + deps_path)

for dep in deps:
    print("Checking dependency: " + dep)
    if dep.startswith("//") and not dep.startswith("//third_party"):
        # Carbon code is always allowed.
        continue
    if dep.startswith("@bazel_tools//"):
        # Internal Bazel dependencies that don't bring in any risky code.
        continue
    if dep.startswith("@llvm-project//"):
        # LLVM itself is fine to depend on as it has the same license as Carbon.
        continue
    if dep.startswith("@llvm_terminfo//") or dep.startswith("@llvm_zlib//"):
        # These are stubs wrapping system libraries for LLVM. They aren't
        # distributed and so should be fine.
        continue
    if (
        dep.startswith("@com_google_absl//")
        or dep.startswith("@com_google_googletest//")
        or dep.startswith("@com_github_google_benchmark//")
    ):
        # This should never be reached from non-test code, but these targets do
        # exist. Specially diagnose them to try to provide a more helpful
        # message.
        sys.exit("ERROR: dependency only allowed in test code: " + dep)
    else:
        # Conservatively fail if a dependency isn't explicitly allowed above.
        sys.exit("ERROR: unknown dependency: " + dep)
