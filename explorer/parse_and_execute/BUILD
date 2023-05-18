# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = [
    "//explorer:__pkg__",
    "//explorer/fuzzing:__pkg__",
])

cc_library(
    name = "parse_and_execute",
    srcs = ["parse_and_execute.cpp"],
    hdrs = ["parse_and_execute.h"],
    deps = [
        "//common:check",
        "//common:error",
        "//explorer/common:trace_stream",
        "//explorer/interpreter:exec_program",
        "//explorer/syntax",
        "//explorer/syntax:prelude",
        "@llvm-project//llvm:Support",
    ],
)
