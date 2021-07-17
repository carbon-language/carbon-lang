# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//executable_semantics:__subpackages__"])

cc_library(
    name = "interpreter",
    srcs = [
        "interpreter.cpp",
    ],
    hdrs = [
        "interpreter.h",
    ],
    deps = [
        ":action",
        ":address",
        ":containers",
        ":frame",
        ":heap",
        ":value",
        "//common:check",
        "//executable_semantics:tracing_flag",
        "//executable_semantics/ast:declaration",
        "//executable_semantics/ast:expression",
        "//executable_semantics/ast:function_definition",
    ],
)

cc_library(
    name = "typecheck",
    srcs = ["typecheck.cpp"],
    hdrs = ["typecheck.h"],
    deps = [
        ":containers",
        ":interpreter",
        "//executable_semantics:tracing_flag",
        "//executable_semantics/ast:expression",
        "//executable_semantics/ast:function_definition",
        "//executable_semantics/ast:statement",
    ],
)

cc_library(
    name = "action",
    srcs = ["action.cpp"],
    hdrs = ["action.h"],
    deps = [
        ":containers",
        ":value",
        "//executable_semantics/ast:expression",
        "//executable_semantics/ast:function_definition",
        "//executable_semantics/ast:statement",
    ],
)

cc_library(
    name = "frame",
    srcs = ["frame.cpp"],
    hdrs = ["frame.h"],
    deps = [
        ":action",
        ":address",
        ":containers",
    ],
)

cc_library(
    name = "value",
    srcs = ["value.cpp"],
    hdrs = ["value.h"],
    deps = [
        ":address",
        ":containers",
        ":field_path",
        "//executable_semantics/ast:statement",
    ],
)

cc_library(
    name = "field_path",
    hdrs = ["field_path.h"],
)

cc_library(
    name = "address",
    hdrs = ["address.h"],
    deps = [
        ":field_path",
    ],
)

cc_library(
    name = "heap",
    srcs = ["heap.cpp"],
    hdrs = ["heap.h"],
    deps = [
        ":address",
        ":value",
    ],
)

cc_library(
    name = "containers",
    srcs = [
        "list_node.h",
    ],
    hdrs = [
        "dictionary.h",
        "stack.h",
    ],
    deps = ["//common:check"],
)
