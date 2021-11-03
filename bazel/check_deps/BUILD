# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@mypy_integration//:mypy.bzl", "mypy_test")

# The filegroups establishing root rules for `genquery` invocations can be
# updated by running:
#
# ```
# ./bazel/check_deps/update_roots.py
# ```
filegroup(
    name = "non_test_cc_rules",
    data = [
        "//executable_semantics",
        "//migrate_cpp/cpp_refactoring",
        "//toolchain/diagnostics:null_diagnostics",
        "//toolchain/driver:carbon",
    ],
)

genquery(
    name = "non_test_cc_deps.txt",
    expression = "kind('cc.* rule', deps(//bazel/check_deps:non_test_cc_rules))",
    opts = [
        "--notool_deps",
        "--noimplicit_deps",
    ],
    scope = [":non_test_cc_rules"],
)

py_test(
    name = "check_non_test_cc_deps",
    srcs = ["check_non_test_cc_deps.py"],
    data = [":non_test_cc_deps.txt"],
    main = "check_non_test_cc_deps.py",
)

mypy_test(
    name = "check_non_test_cc_deps_mypy_test",
    include_imports = True,
    deps = [":check_non_test_cc_deps"],
)

# Note that this script expects to be run directly and not via Bazel as it in
# turn invokes Bazel.
py_binary(
    name = "update_roots",
    srcs = ["update_roots.py"],
)

mypy_test(
    name = "update_roots_mypy_test",
    include_imports = True,
    deps = [":update_roots"],
)
