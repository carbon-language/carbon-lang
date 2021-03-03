# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//executable_semantics:__subpackages__"])

# TODO: It may be helpful to break this apart.
cc_library(
    name = "interpreter",
    srcs = [
        "action.cpp",
        "interpreter.cpp",
        "typecheck.cpp",
        "value.cpp",
    ],
    hdrs = [
        "action.h",
        "assoc_list.h",
        "cons_list.h",
        "interpreter.h",
        "stack.h",
        "typecheck.h",
        "value.h",
    ],
    deps = [
        "//executable_semantics:syntax_helpers_hdrs",
        "//executable_semantics/ast:declaration",
        "//executable_semantics/ast:expression",
        "//executable_semantics/ast:function_definition",
        "//executable_semantics/ast:member",
        "//executable_semantics/ast:statement",
    ],
)
