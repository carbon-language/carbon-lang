# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//executable_semantics:__pkg__"])

# These currently have to be a single build rule because of a dependency cycle
# in printing.
cc_library(
    name = "action_and_value",
    srcs = [
        "action.cpp",
        "value.cpp",
    ],
    hdrs = [
        "action.h",
        "value.h",
    ],
    deps = [
        ":address",
        ":dictionary",
        ":field_path",
        ":heap_allocation_interface",
        ":stack",
        "//common:ostream",
        "//executable_semantics/ast:declaration",
        "//executable_semantics/ast:expression",
        "//executable_semantics/ast:statement",
        "//executable_semantics/common:arena",
        "//executable_semantics/common:error",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "action_stack",
    srcs = ["action_stack.cpp"],
    hdrs = ["action_stack.h"],
    deps = [
        ":action_and_value",
        ":stack",
        "//common:ostream",
        "//executable_semantics/ast:statement",
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
    name = "exec_program",
    srcs = ["exec_program.cpp"],
    hdrs = ["exec_program.h"],
    deps = [
        ":interpreter",
        ":resolve_control_flow",
        ":resolve_names",
        ":type_checker",
        "//executable_semantics/ast",
        "//executable_semantics/syntax",
    ],
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
        ":action_and_value",
        ":address",
        ":heap_allocation_interface",
        "//common:ostream",
        "//executable_semantics/ast:source_location",
        "//executable_semantics/common:nonnull",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "heap_allocation_interface",
    hdrs = ["heap_allocation_interface.h"],
    deps = [
        ":address",
        "//executable_semantics/common:nonnull",
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
        ":action_and_value",
        ":action_stack",
        ":address",
        ":heap",
        "//common:check",
        "//common:ostream",
        "//executable_semantics/ast:declaration",
        "//executable_semantics/ast:expression",
        "//executable_semantics/common:arena",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "resolve_control_flow",
    srcs = ["resolve_control_flow.cpp"],
    hdrs = ["resolve_control_flow.h"],
    deps = [
        "//common:check",
        "//executable_semantics/ast",
        "//executable_semantics/ast:declaration",
        "//executable_semantics/ast:statement",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "resolve_names",
    srcs = ["resolve_names.cpp"],
    hdrs = ["resolve_names.h"],
    deps = [
        "//common:check",
        "//executable_semantics/ast",
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
        "//executable_semantics/ast",
        "//executable_semantics/ast:declaration",
        "//executable_semantics/ast:expression",
        "//executable_semantics/ast:statement",
        "//executable_semantics/common:arena",
        "@llvm-project//llvm:Support",
    ],
)
