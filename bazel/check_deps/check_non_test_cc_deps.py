#!/usr/bin/env python3

"""Check that non-test C++ rules only depend on Carbon and LLVM.

Carbon works to ensure its user-visible libraries and binaries only depend on
their code and LLVM. Among other benefits, this provides a single, simple
license used for the whole project.

However, we frequently use third-party projects and libraries where useful in
our test code. Here, we verify that the dependencies of non-test C++ rules only
include Carbon and LLVM code.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import sys
from pathlib import Path

runfiles = Path(os.environ["TEST_SRCDIR"])
deps_path = runfiles / "_main" / "bazel" / "check_deps" / "non_test_cc_deps.txt"
try:
    with deps_path.open() as deps_file:
        deps = deps_file.read().splitlines()
except FileNotFoundError:
    sys.exit("ERROR: unable to find deps file: %s" % deps_path)

# This errors out on dependencies that aren't recognized, and continues on
# allowed dependencies.
for dep in deps:
    print("Checking dependency: " + dep)
    repo, _, rule = dep.partition("//")

    if repo == "@@+llvm_project+llvm-project":
        package, _, rule = rule.partition(":")

        # Other packages in the LLVM project shouldn't be accidentally used
        # in Carbon. We can expand the above list if use cases emerge.
        if package not in (
            "llvm",
            "lld",
            "clang",
            "clang-tools-extra/clangd",
            "libunwind",
        ):
            sys.exit(
                "ERROR: unexpected dependency into the LLVM project: %s" % dep
            )

        # Check for accidentally using the copy of GoogleTest in LLVM.
        if rule in ("gmock", "gtest", "gtest_main"):
            sys.exit(
                "ERROR: dependency on LLVM's GoogleTest from non-test code: %s"
                % dep
            )

        # The rest of LLVM, LLD, and Clang themselves are safe to depend on.
        continue

    # Carbon code is always allowed.
    if repo == "" and not rule.startswith("third_party"):
        continue

    # An empty stub library added by rules_cc:
    # https://github.com/bazelbuild/rules_cc/blob/main/BUILD
    if repo == "@@rules_cc+" and rule == ":link_extra_lib":
        continue

    # An utility library provided by Bazel that is under a compatible license.
    if repo == "@@bazel_tools" and rule == "tools/cpp/runfiles:runfiles":
        continue

    # These are stubs wrapping system libraries for LLVM. They aren't
    # distributed and so should be fine.
    if repo in (
        "@@zlib+",
        "@@zstd+",
    ):
        continue

    # This should never be reached from non-test code, but these targets do
    # exist. Specially diagnose them to try to provide a more helpful
    # message.
    if repo in (
        "@google_benchmark",
        "@abseil-cpp",
        "@googletest",
    ):
        sys.exit("ERROR: dependency only allowed in test code: %s" % dep)

    # Conservatively fail if a dependency isn't explicitly allowed above.
    sys.exit(f"ERROR: unknown dependency on {repo}: {dep}")
