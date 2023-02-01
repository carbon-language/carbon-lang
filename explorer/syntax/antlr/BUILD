# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/antlr:defs.bzl", "antlr_cc_library")

package(default_visibility = [
    "//explorer/syntax:__pkg__",
])

antlr_cc_library(
    name = "antlr",
    src = "Carbon.g4",
    package = "Carbon::Antlr",
)

cc_library(
    name = "parse",
    srcs = [
        "parse.cpp",
        "parser_visitor.cpp",
        "parser_visitor.h",
    ],
    hdrs = ["parse.h"],
    deps = [
        ":antlr",
        "//common:error",
        "//explorer/ast",
        "//explorer/common:arena",
        "@antlr4_runtimes//:cpp",
    ],
)
