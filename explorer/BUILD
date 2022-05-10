# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@mypy_integration//:mypy.bzl", "mypy_test")

package(default_visibility = [
    "//bazel/check_deps:__pkg__",
    "//explorer:__subpackages__",
])

filegroup(
    name = "standard_libraries",
    srcs = ["data/prelude.carbon"],
)

cc_binary(
    name = "explorer",
    srcs = ["main.cpp"],
    data = [":standard_libraries"],
    deps = [
        "//common:error",
        "//explorer/common:arena",
        "//explorer/common:nonnull",
        "//explorer/interpreter:exec_program",
        "//explorer/syntax",
        "@llvm-project//llvm:Support",
    ],
)

py_binary(
    name = "gen_rtti",
    srcs = ["gen_rtti.py"],
)

# This script should be run directly; the target is only provided for the
# mypy_test.
py_binary(
    name = "update_checks_mypy_wrapper",
    srcs = ["update_checks.py"],
    main = "update_checks.py",
)

mypy_test(
    name = "update_checks_mypy_test",
    include_imports = True,
    deps = [":update_checks_mypy_wrapper"],
)
