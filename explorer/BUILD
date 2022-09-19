# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/macos_malloc:env.bzl", "macos_malloc_env")

package(default_visibility = [
    "//bazel/check_deps:__pkg__",
    "//explorer:__subpackages__",
    "//installers:__subpackages__",
])

filegroup(
    name = "standard_libraries",
    srcs = ["data/prelude.carbon"],
)

cc_library(
    name = "main",
    srcs = ["main.cpp"],
    hdrs = ["main.h"],
    deps = [
        "//common:error",
        "//explorer/common:arena",
        "//explorer/common:nonnull",
        "//explorer/interpreter:exec_program",
        "//explorer/syntax",
        "//explorer/syntax:prelude",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "explorer",
    srcs = ["main_bin.cpp"],
    deps = [
        ":main",
        "@llvm-project//llvm:Support",
    ],
    env = macos_malloc_env(),
)

py_binary(
    name = "gen_rtti",
    srcs = ["gen_rtti.py"],
)
