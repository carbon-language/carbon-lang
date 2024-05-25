# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")
load("//bazel/sh_run:rules.bzl", "glob_sh_run")
load("//testing/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    data = glob(["testdata/**/*.carbon"]),
)

cc_library(
    name = "node_stack",
    srcs = ["node_stack.cpp"],
    hdrs = ["node_stack.h"],
    deps = [
        "//common:check",
        "//common:ostream",
        "//common:vlog",
        "//toolchain/parse:node_kind",
        "//toolchain/parse:tree",
        "//toolchain/sem_ir:ids",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "scope_stack",
    srcs = ["scope_stack.cpp"],
    hdrs = [
        "lexical_lookup.h",
        "scope_index.h",
        "scope_stack.h",
    ],
    deps = [
        "//common:check",
        "//common:ostream",
        "//common:vlog",
        "//toolchain/base:index_base",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:ids",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "context",
    srcs = [
        "context.cpp",
        "convert.cpp",
        "decl_name_stack.cpp",
        "eval.cpp",
        "function.cpp",
        "import_ref.cpp",
        "inst_block_stack.cpp",
        "merge.cpp",
        "modifiers.cpp",
        "return.cpp",
        "subst.cpp",
    ],
    hdrs = [
        "context.h",
        "convert.h",
        "decl_name_stack.h",
        "decl_state.h",
        "diagnostic_helpers.h",
        "eval.h",
        "function.h",
        "import_ref.h",
        "inst_block_stack.h",
        "merge.h",
        "modifiers.h",
        "param_and_arg_refs_stack.h",
        "pending_block.h",
        "return.h",
        "subst.h",
    ],
    deps = [
        ":node_stack",
        "//common:check",
        "//common:vlog",
        "//toolchain/base:index_base",
        "//toolchain/base:kind_switch",
        "//toolchain/check:scope_stack",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse:node_kind",
        "//toolchain/parse:tree",
        "//toolchain/parse:tree_node_diagnostic_converter",
        "//toolchain/sem_ir:builtin_kind",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:ids",
        "//toolchain/sem_ir:inst",
        "//toolchain/sem_ir:inst_kind",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "check",
    srcs = [
        "check.cpp",
        "handle.h",
    ] +
    # Glob handler files to avoid missing any.
    glob([
        "handle_*.cpp",
    ]),
    hdrs = ["check.h"],
    deps = [
        ":call",
        ":context",
        ":impl",
        ":import",
        ":interface",
        ":member_access",
        ":operator",
        ":pointer_dereference",
        "//common:check",
        "//common:error",
        "//common:ostream",
        "//common:variant_helpers",
        "//common:vlog",
        "//toolchain/base:kind_switch",
        "//toolchain/base:pretty_stack_trace_function",
        "//toolchain/base:value_store",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse:node_kind",
        "//toolchain/parse:tree",
        "//toolchain/parse:tree_node_diagnostic_converter",
        "//toolchain/sem_ir:entry_point",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:ids",
        "//toolchain/sem_ir:inst",
        "//toolchain/sem_ir:inst_kind",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "check_fuzzer",
    size = "small",
    srcs = ["check_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/*"]),
    deps = [
        "//toolchain/driver",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "call",
    srcs = ["call.cpp"],
    hdrs = ["call.h"],
    deps = [
        ":context",
        "//common:check",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:ids",
        "//toolchain/sem_ir:inst",
        "//toolchain/sem_ir:inst_kind",
    ],
)

cc_library(
    name = "impl",
    srcs = ["impl.cpp"],
    hdrs = ["impl.h"],
    deps = [
        ":context",
        "//common:check",
        "//toolchain/base:kind_switch",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:ids",
        "//toolchain/sem_ir:inst",
        "//toolchain/sem_ir:inst_kind",
    ],
)

cc_library(
    name = "import",
    srcs = ["import.cpp"],
    hdrs = ["import.h"],
    deps = [
        ":context",
        "//common:check",
        "//toolchain/base:kind_switch",
        "//toolchain/parse:node_kind",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:ids",
        "//toolchain/sem_ir:inst",
        "//toolchain/sem_ir:inst_kind",
    ],
)

cc_library(
    name = "interface",
    srcs = ["interface.cpp"],
    hdrs = ["interface.h"],
    deps = [
        ":context",
        "//common:check",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:ids",
        "//toolchain/sem_ir:inst",
        "//toolchain/sem_ir:inst_kind",
    ],
)

cc_library(
    name = "member_access",
    srcs = ["member_access.cpp"],
    hdrs = ["member_access.h"],
    deps = [
        ":context",
        "//common:check",
        "//toolchain/base:kind_switch",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:ids",
        "//toolchain/sem_ir:inst",
        "//toolchain/sem_ir:inst_kind",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "operator",
    srcs = ["operator.cpp"],
    hdrs = ["operator.h"],
    deps = [
        ":call",
        ":context",
        ":member_access",
        "//common:check",
        "//toolchain/parse:node_kind",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:ids",
        "//toolchain/sem_ir:inst",
        "//toolchain/sem_ir:inst_kind",
    ],
)

cc_library(
    name = "pointer_dereference",
    srcs = ["pointer_dereference.cpp"],
    hdrs = ["pointer_dereference.h"],
    deps = [
        ":context",
        "//common:check",
        "//toolchain/parse:node_kind",
        "//toolchain/sem_ir:ids",
        "//toolchain/sem_ir:inst",
        "//toolchain/sem_ir:inst_kind",
        "@llvm-project//llvm:Support",
    ],
)

glob_sh_run(
    args = [
        "$(location //toolchain/driver:carbon)",
        "compile",
        "--phase=check",
        "--dump-sem-ir",
        "--dump-raw-sem-ir",
    ],
    data = ["//toolchain/driver:carbon"],
    file_exts = ["carbon"],
)

glob_sh_run(
    args = [
        "$(location //toolchain/driver:carbon)",
        "-v",
        "compile",
        "--phase=check",
    ],
    data = ["//toolchain/driver:carbon"],
    file_exts = ["carbon"],
    run_ext = "verbose",
)
