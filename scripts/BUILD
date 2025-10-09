# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "no_op_test",
    size = "small",
    srcs = ["no_op_test.py"],
    main = "no_op_test.py",
)

cc_library(
    name = "deps_for_clangd_tidy",
    srcs = ["deps_for_clangd_tidy.cpp"],
    deps = [
        # `@boost_unordered` uses `strip_prefix`, which results in
        # `_virtual_includes` being part of the `compile_commands.json`. To
        # support tools like `clangd-tidy`, this is intended to generate that directory.
        #
        # For example:
        # bazel-out/<mode>/bin/external/+_repo_rules+boost_unordered/_virtual_includes/boost_unordered
        "@boost_unordered",
    ],
)
