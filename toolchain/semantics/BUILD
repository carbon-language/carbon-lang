# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//bazel/sh_run:rules.bzl", "glob_sh_run")
load("//testing/file_test:rules.bzl", "file_test")
load("//testing/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "semantics_builtin_kind",
    srcs = ["semantics_builtin_kind.cpp"],
    hdrs = ["semantics_builtin_kind.h"],
    textual_hdrs = ["semantics_builtin_kind.def"],
    deps = ["//common:enum_base"],
)

cc_library(
    name = "semantics_node_kind",
    srcs = ["semantics_node_kind.cpp"],
    hdrs = ["semantics_node_kind.h"],
    textual_hdrs = ["semantics_node_kind.def"],
    deps = [
        "//common:enum_base",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "semantics_node",
    srcs = ["semantics_node.cpp"],
    hdrs = ["semantics_node.h"],
    deps = [
        ":semantics_builtin_kind",
        ":semantics_node_kind",
        "//common:check",
        "//common:ostream",
        "//toolchain/base:index_base",
        "//toolchain/parser:parse_tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "semantics_node_stack",
    srcs = ["semantics_node_stack.cpp"],
    hdrs = ["semantics_node_stack.h"],
    deps = [
        ":semantics_node",
        "//common:check",
        "//common:ostream",
        "//common:vlog",
        "//toolchain/parser:parse_node_kind",
        "//toolchain/parser:parse_tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "check",
    srcs = [
        "check.cpp",
        "semantics_context.cpp",
        "semantics_declaration_name_stack.cpp",
        "semantics_node_block_stack.cpp",
    ] +
    # Glob handler files to avoid missing any.
    glob([
        "semantics_handle*.cpp",
    ]),
    hdrs = [
        "check.h",
        "semantics_context.h",
        "semantics_declaration_name_stack.h",
        "semantics_node_block_stack.h",
    ],
    deps = [
        ":semantics_builtin_kind",
        ":semantics_ir",
        ":semantics_node",
        ":semantics_node_kind",
        ":semantics_node_stack",
        "//common:check",
        "//common:ostream",
        "//common:vlog",
        "//toolchain/base:pretty_stack_trace_function",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/diagnostics:diagnostic_kind",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_node_kind",
        "//toolchain/parser:parse_tree",
        "//toolchain/parser:parse_tree_node_location_translator",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "semantics_ir",
    srcs = ["semantics_ir.cpp"],
    hdrs = ["semantics_ir.h"],
    deps = [
        ":semantics_builtin_kind",
        ":semantics_node",
        ":semantics_node_kind",
        "//common:check",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "semantics_ir_formatter",
    srcs = ["semantics_ir_formatter.cpp"],
    hdrs = ["semantics_ir_formatter.h"],
    deps = [
        ":semantics_ir",
        ":semantics_node_kind",
        "//toolchain/lexer:tokenized_buffer",
        "//toolchain/parser:parse_tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "semantics_ir_test",
    size = "small",
    srcs = ["semantics_ir_test.cpp"],
    deps = [
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "//toolchain/base:yaml_test_helpers",
        "//toolchain/driver",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

file_test(
    name = "semantics_file_test",
    srcs = ["semantics_file_test.cpp"],
    tests = glob(["testdata/**/*.carbon"]),
    deps = [
        "//toolchain/driver:driver_file_test_base",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "semantics_fuzzer",
    size = "small",
    srcs = ["semantics_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/*"]),
    deps = [
        "//toolchain/driver",
        "@llvm-project//llvm:Support",
    ],
)

glob_sh_run(
    args = [
        "$(location //toolchain/driver:carbon)",
        "dump",
        "semantics-ir",
    ],
    data = ["//toolchain/driver:carbon"],
    file_exts = ["carbon"],
)

glob_sh_run(
    args = [
        "$(location //toolchain/driver:carbon)",
        "-v",
        "dump",
        "semantics-ir",
    ],
    data = ["//toolchain/driver:carbon"],
    file_exts = ["carbon"],
    run_ext = "verbose",
)
