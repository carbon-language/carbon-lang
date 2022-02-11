# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@mypy_integration//:mypy.bzl", "mypy_test")

py_library(
    name = "utils",
    srcs = ["utils.py"],
)

# Note that this Python script is intended to be run directly, and not through
# Bazel. We have a rule for it so we can type check it.
py_binary(
    name = "buildifier",
    srcs = ["buildifier.py"],
    deps = [":utils"],
)

mypy_test(
    name = "buildifier_mypy_test",
    include_imports = True,
    deps = [":buildifier"],
)

# Note that this Python script is intended to be run directly, and not through
# Bazel. We have a rule for it so we can type check it.
py_binary(
    name = "create_compdb",
    srcs = ["create_compdb.py"],
    deps = [":utils"],
)

mypy_test(
    name = "create_compdb_mypy_test",
    include_imports = True,
    deps = [":create_compdb"],
)

# Note that this Python script is intended to be run directly, and not through
# Bazel. We have a rule for it so we can type check it.
py_binary(
    name = "fix_cc_deps",
    srcs = ["fix_cc_deps.py"],
    deps = [":utils"],
)

mypy_test(
    name = "fix_cc_deps_mypy_test",
    include_imports = True,
    deps = [":fix_cc_deps"],
)

# Note that this Python script is intended to be run directly, and not through
# Bazel. We have a rule for it so we can type check it.
py_binary(
    name = "forbid_llvm_googletest",
    srcs = ["forbid_llvm_googletest.py"],
    deps = [":utils"],
)

mypy_test(
    name = "forbid_llvm_googletest_mypy_test",
    include_imports = True,
    deps = [":forbid_llvm_googletest"],
)
