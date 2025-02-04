# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_python//python:defs.bzl", "py_test")

# The filegroups establishing root rules for `genquery` invocations can be
# updated by running:
#
# ```
# ./bazel/check_deps/update_roots.py
# ```
filegroup(
    name = "non_test_cc_rules",
    data = [
        "//explorer",
        "//installers/local:carbon",
        "//toolchain/install:carbon-busybox",
        # The tree sitter rules can't be queried; evaluation fails on
        # @platforms.
        # "//utils/tree_sitter",
    ],
    tags = ["manual"],
)

genquery(
    name = "non_test_cc_deps.txt",
    expression =
        "kind('cc.* rule', deps(//bazel/check_deps:non_test_cc_rules))",
    opts = [
        "--notool_deps",
        "--noimplicit_deps",
    ],
    scope = [":non_test_cc_rules"],
)

py_test(
    name = "check_non_test_cc_deps",
    size = "small",
    srcs = ["check_non_test_cc_deps.py"],
    data = [":non_test_cc_deps.txt"],
    main = "check_non_test_cc_deps.py",
)
