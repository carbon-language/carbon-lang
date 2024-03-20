# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//explorer/parse_and_execute:__pkg__"])

cc_library(
    name = "action",
    srcs = [
        "action.cpp",
    ],
    hdrs = [
        "action.h",
    ],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":dictionary",
        ":heap_allocation_interface",
        ":stack",
        "//common:check",
        "//common:error",
        "//common:ostream",
        "//explorer/ast",
        "//explorer/base:arena",
        "//explorer/base:error_builders",
        "//explorer/base:nonnull",
        "//explorer/base:print_as_id",
        "//explorer/base:source_location",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "action_stack",
    srcs = [
        "action_stack.cpp",
    ],
    hdrs = [
        "action_stack.h",
    ],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":action",
        ":stack",
        "//common:check",
        "//common:error",
        "//common:ostream",
        "//explorer/ast",
        "//explorer/base:trace_stream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "dictionary",
    hdrs = ["dictionary.h"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = ["//explorer/base:arena"],
)

cc_library(
    name = "exec_program",
    srcs = ["exec_program.cpp"],
    hdrs = ["exec_program.h"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":interpreter",
        ":resolve_control_flow",
        ":resolve_names",
        ":resolve_unformed",
        ":type_checker",
        "//common:check",
        "//common:error",
        "//common:ostream",
        "//explorer/ast",
        "//explorer/base:arena",
        "//explorer/base:trace_stream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "heap",
    srcs = ["heap.cpp"],
    hdrs = ["heap.h"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":action",
        ":heap_allocation_interface",
        "//common:check",
        "//common:error",
        "//common:ostream",
        "//explorer/ast",
        "//explorer/base:error_builders",
        "//explorer/base:nonnull",
        "//explorer/base:source_location",
        "//explorer/base:trace_stream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "heap_allocation_interface",
    hdrs = ["heap_allocation_interface.h"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        "//common:error",
        "//explorer/ast",
        "//explorer/base:arena",
        "//explorer/base:nonnull",
        "//explorer/base:source_location",
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
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":action",
        ":action_stack",
        ":heap",
        ":pattern_match",
        ":stack",
        ":type_utils",
        "//common:check",
        "//common:error",
        "//common:ostream",
        "//explorer/ast",
        "//explorer/ast:expression_category",
        "//explorer/base:arena",
        "//explorer/base:error_builders",
        "//explorer/base:print_as_id",
        "//explorer/base:source_location",
        "//explorer/base:trace_stream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "resolve_control_flow",
    srcs = ["resolve_control_flow.cpp"],
    hdrs = ["resolve_control_flow.h"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        "//common:check",
        "//explorer/ast",
        "//explorer/base:error_builders",
        "//explorer/base:nonnull",
        "//explorer/base:print_as_id",
        "//explorer/base:trace_stream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "resolve_names",
    srcs = ["resolve_names.cpp"],
    hdrs = ["resolve_names.h"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":stack_space",
        "//common:check",
        "//explorer/ast",
        "//explorer/ast:static_scope",
        "//explorer/base:arena",
        "//explorer/base:print_as_id",
        "//explorer/base:trace_stream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "stack",
    hdrs = ["stack.h"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
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
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":action",
        "//common:error",
        "//explorer/ast",
        "//explorer/base:nonnull",
        "@llvm-project//llvm:Support",
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
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    textual_hdrs = [
        "builtins.def",
    ],
    deps = [
        ":action",
        ":dictionary",
        ":interpreter",
        ":pattern_analysis",
        ":pattern_match",
        ":stack_space",
        ":type_structure",
        ":type_utils",
        "//common:check",
        "//common:enum_base",
        "//common:error",
        "//common:ostream",
        "//explorer/ast",
        "//explorer/base:arena",
        "//explorer/base:error_builders",
        "//explorer/base:nonnull",
        "//explorer/base:print_as_id",
        "//explorer/base:source_location",
        "//explorer/base:trace_stream",
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
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":stack_space",
        "//common:check",
        "//explorer/ast",
        "//explorer/ast:static_scope",
        "//explorer/base:error_builders",
        "//explorer/base:nonnull",
        "//explorer/base:print_as_id",
        "//explorer/base:trace_stream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "stack_space",
    srcs = ["stack_space.cpp"],
    hdrs = ["stack_space.h"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        "//common:check",
        "//common:error",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "type_structure",
    srcs = [
        "type_structure.cpp",
    ],
    hdrs = [
        "type_structure.h",
    ],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        "//common:ostream",
        "//explorer/ast",
        "//explorer/ast:expression_category",
        "//explorer/base:nonnull",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "type_utils",
    srcs = [
        "type_utils.cpp",
    ],
    hdrs = [
        "type_utils.h",
    ],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        "//explorer/ast",
        "//explorer/base:nonnull",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "pattern_match",
    srcs = [
        "pattern_match.cpp",
    ],
    hdrs = [
        "pattern_match.h",
    ],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":action",
        ":type_utils",
        "//explorer/ast",
        "//explorer/base:arena",
        "//explorer/base:nonnull",
        "//explorer/base:source_location",
        "//explorer/base:trace_stream",
        "@llvm-project//llvm:Support",
    ],
)
