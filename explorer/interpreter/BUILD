# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//explorer:__pkg__"])

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
        "//common:check",
        "//common:ostream",
        "//explorer/ast",
        "//explorer/common:arena",
        "//explorer/common:error_builders",
        "//explorer/common:nonnull",
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
        "//common:error",
        "//common:ostream",
        "//explorer/ast",
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
    deps = ["//explorer/common:arena"],
)

cc_library(
    name = "exec_program",
    srcs = ["exec_program.cpp"],
    hdrs = ["exec_program.h"],
    visibility = [
        "//explorer:__pkg__",
        "//explorer/fuzzing:__pkg__",
    ],
    deps = [
        ":interpreter",
        ":resolve_control_flow",
        ":resolve_names",
        ":resolve_unformed",
        ":type_checker",
        "//common:check",
        "//common:ostream",
        "//explorer/ast",
        "//explorer/common:arena",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "field_path",
    hdrs = ["field_path.h"],
    deps = [
        "//common:ostream",
        "//explorer/ast",
        "//explorer/ast:static_scope",
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
        "//explorer/common:error_builders",
        "//explorer/common:nonnull",
        "//explorer/common:source_location",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "heap_allocation_interface",
    hdrs = ["heap_allocation_interface.h"],
    deps = [
        ":address",
        "//explorer/common:arena",
        "//explorer/common:nonnull",
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
        ":stack",
        "//common:check",
        "//common:ostream",
        "//explorer/ast",
        "//explorer/common:arena",
        "//explorer/common:error_builders",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "resolve_control_flow",
    srcs = ["resolve_control_flow.cpp"],
    hdrs = ["resolve_control_flow.h"],
    deps = [
        "//common:check",
        "//explorer/ast",
        "//explorer/common:error_builders",
        "//explorer/common:nonnull",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "resolve_names",
    srcs = ["resolve_names.cpp"],
    hdrs = ["resolve_names.h"],
    deps = [
        ":action_and_value",
        "//common:check",
        "//explorer/ast",
        "//explorer/ast:static_scope",
        "//explorer/common:arena",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "stack",
    hdrs = ["stack.h"],
    deps = ["//common:check"],
)

cc_library(
    name = "pattern_analysis",
    srcs = [
        "pattern_analysis.cpp",
    ],
    hdrs = [
        "pattern_analysis.h",
    ],
    deps = [
        ":action_and_value",
        "//common:error",
        "//explorer/ast",
        "//explorer/common:nonnull",
        "@llvm-project//llvm:Support",
    ],
)
cc_library(
    name = "type_checker",
    srcs = [
        "builtins.cpp",
        "impl_scope.cpp",
        "type_checker.cpp",
    ],
    hdrs = [
        "builtins.h",
        "impl_scope.h",
        "type_checker.h",
    ],
    deps = [
        ":action_and_value",
        ":dictionary",
        ":interpreter",
        ":pattern_analysis",
        "//common:error",
        "//common:ostream",
        "//explorer/ast",
        "//explorer/common:arena",
        "//explorer/common:error_builders",
        "//explorer/common:nonnull",
        "//explorer/common:source_location",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "resolve_unformed",
    srcs = [
        "resolve_unformed.cpp",
    ],
    hdrs = [
        "resolve_unformed.h",
    ],
    deps = [
        "//common:check",
        "//explorer/ast",
        "//explorer/ast:static_scope",
        "//explorer/common:error_builders",
        "//explorer/common:nonnull",
        "@llvm-project//llvm:Support",
    ],
)
