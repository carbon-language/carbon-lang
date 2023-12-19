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

# bazel_dep repos can have a `name~version` format, whereas use_repo and
# use_repo_rule have a `@@_main~rule_name~repo_name` format. We process the
# latter case first because we only have a couple; once done, we can assume
# anything after the ~ is a version.
for dep in deps:
    print("Checking dependency: " + dep)
    repo, _, rule = dep.partition("//")

    if repo == "@@_main~llvm_project~llvm-project":
        package, _, rule = rule.partition(":")

        # Other packages in the LLVM project shouldn't be accidentally used
        # in Carbon. We can expand the above list if use cases emerge.
        if package not in ("llvm", "lld", "clang", "clang-tools-extra/clangd"):
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

    # Ignore the version, just use the repo name.
    repo_base = repo.split("~")[0]

    # Carbon code is always allowed.
    if repo_base == "" and not rule.startswith("third_party"):
        continue

    # An empty stub library added by rules_cc:
    # https://github.com/bazelbuild/rules_cc/blob/main/BUILD
    if repo_base == "@@rules_cc" and rule == ":link_extra_lib":
        continue

    # These are stubs wrapping system libraries for LLVM. They aren't
    # distributed and so should be fine.
    if repo_base in (
        "@@zlib",
        "@@zstd",
    ):
        continue

    # This should never be reached from non-test code, but these targets do
    # exist. Specially diagnose them to try to provide a more helpful
    # message.
    if repo_base in (
        "@com_github_google_benchmark",
        "@com_github_protocolbuffers_protobuf",
        "@com_google_absl",
        "@com_google_googletest",
    ):
        sys.exit("ERROR: dependency only allowed in test code: %s" % dep)

    # Conservatively fail if a dependency isn't explicitly allowed above.
    sys.exit("ERROR: unknown dependency: %s" % dep)
