# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@mypy_integration//:mypy.bzl", "mypy_test")

py_library(
    name = "scripts_utils",
    srcs = ["scripts_utils.py"],
)

# Note that this Python script is intended to be run directly, and not through
# Bazel. We have a rule for it so we can type check it.
py_binary(
    name = "create_compdb",
    testonly = 1,
    srcs = ["create_compdb.py"],
    deps = [":scripts_utils"],
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
    testonly = 1,
    srcs = ["fix_cc_deps.py"],
    deps = [":scripts_utils"],
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
    testonly = 1,
    srcs = ["forbid_llvm_googletest.py"],
    deps = [":scripts_utils"],
)

mypy_test(
    name = "forbid_llvm_googletest_mypy_test",
    include_imports = True,
    deps = [":forbid_llvm_googletest"],
)

# Note that this Python script is intended to be run directly, and not through
# Bazel. We have a rule for it so we can type check it.
py_binary(
    name = "run_bazel",
    testonly = 1,
    srcs = ["run_bazel.py"],
    deps = [":scripts_utils"],
)

mypy_test(
    name = "run_bazel_mypy_test",
    include_imports = True,
    deps = [":run_bazel"],
)

# Note that this Python script is intended to be run directly, and not through
# Bazel. We have a rule for it so we can type check it.
py_binary(
    name = "run_buildifier",
    testonly = 1,
    srcs = ["run_buildifier.py"],
    deps = [":scripts_utils"],
)

mypy_test(
    name = "run_buildifier_mypy_test",
    include_imports = True,
    deps = [":run_buildifier"],
)

# Note that this Python script is intended to be run directly, and not through
# Bazel. We have a rule for it so we can type check it.
py_binary(
    name = "run_buildozer",
    testonly = 1,
    srcs = ["run_buildozer.py"],
    deps = [":scripts_utils"],
)

mypy_test(
    name = "run_buildozer_mypy_test",
    include_imports = True,
    deps = [":run_buildozer"],
)
