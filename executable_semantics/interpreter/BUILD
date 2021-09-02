# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//executable_semantics:__subpackages__"])

# These must be a single build rule because of a dependency cycle in printing.
cc_library(
    name = "action_frame_and_value",
    srcs = [
        "action.cpp",
        "frame.cpp",
        "value.cpp",
    ],
    hdrs = [
        "action.h",
        "frame.h",
        "value.h",
    ],
    deps = [
        ":address",
        ":dictionary",
        ":field_path",
        ":stack",
        "//common:ostream",
        "//executable_semantics/ast:expression",
        "//executable_semantics/ast:function_definition",
        "//executable_semantics/ast:statement",
        "//executable_semantics/common:arena",
        "//executable_semantics/common:error",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "address",
    hdrs = ["address.h"],
    deps = [
        ":field_path",
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "dictionary",
    hdrs = ["dictionary.h"],
    deps = ["//executable_semantics/common:arena"],
)

cc_library(
    name = "field_path",
    hdrs = ["field_path.h"],
    deps = [
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "heap",
    srcs = ["heap.cpp"],
    hdrs = ["heap.h"],
    deps = [
        ":action_frame_and_value",
        ":address",
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "interpreter",
    srcs = [
        "interpreter.cpp",
    ],
    hdrs = [
        "interpreter.h",
    ],
    deps = [
        ":action_frame_and_value",
        ":address",
        ":heap",
        "//common:check",
        "//common:ostream",
        "//executable_semantics/ast:declaration",
        "//executable_semantics/ast:expression",
        "//executable_semantics/ast:function_definition",
        "//executable_semantics/common:arena",
        "//executable_semantics/common:tracing_flag",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "stack",
    hdrs = ["stack.h"],
    deps = ["//common:check"],
)

cc_library(
    name = "type_checker",
    srcs = ["type_checker.cpp"],
    hdrs = ["type_checker.h"],
    deps = [
        ":dictionary",
        ":interpreter",
        "//common:ostream",
        "//executable_semantics/ast:expression",
        "//executable_semantics/ast:function_definition",
        "//executable_semantics/ast:statement",
        "//executable_semantics/common:arena",
        "//executable_semantics/common:tracing_flag",
        "@llvm-project//llvm:Support",
    ],
)
