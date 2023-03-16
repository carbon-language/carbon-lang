# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//explorer:__pkg__"])

cc_library(
    name = "action",
    srcs = [
        "action.cpp",
    ],
    hdrs = [
        "action.h",
    ],
    deps = [
        ":dictionary",
        ":heap_allocation_interface",
        ":stack",
        "//common:check",
        "//common:error",
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
    srcs = [
        "action_stack.cpp",
        "stack_fragment.cpp",
    ],
    hdrs = [
        "action_stack.h",
        "stack_fragment.h",
    ],
    deps = [
        ":action",
        ":stack",
        "//common:check",
        "//common:error",
        "//common:ostream",
        "//explorer/ast",
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
        ":trace_stream",
        ":type_checker",
        "//common:check",
        "//common:ostream",
        "//explorer/ast",
        "//explorer/common:arena",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "heap",
    srcs = ["heap.cpp"],
    hdrs = ["heap.h"],
    deps = [
        ":action",
        ":heap_allocation_interface",
        "//common:ostream",
        "//explorer/ast",
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
        "//explorer/ast",
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
        ":action",
        ":action_stack",
        ":heap",
        ":stack",
        ":trace_stream",
        "//common:check",
        "//common:error",
        "//common:ostream",
        "//explorer/ast",
        "//explorer/common:arena",
        "//explorer/common:error_builders",
        "//explorer/common:source_location",
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
        ":action",
        "//common:error",
        "//explorer/ast",
        "//explorer/common:nonnull",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "trace_stream",
    hdrs = ["trace_stream.h"],
    visibility = [
        "//explorer:__pkg__",
        "//explorer/fuzzing:__pkg__",
    ],
    deps = [
        "//common:check",
        "//common:ostream",
        "//explorer/common:nonnull",
    ],
)

cc_library(
    name = "type_checker",
    srcs = [
        "builtins.cpp",
        "impl_scope.cpp",
        "matching_impl_set.cpp",
        "type_checker.cpp",
    ],
    hdrs = [
        "builtins.h",
        "impl_scope.h",
        "matching_impl_set.h",
        "type_checker.h",
    ],
    deps = [
        ":action",
        ":dictionary",
        ":interpreter",
        ":pattern_analysis",
        ":trace_stream",
        "//common:check",
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
