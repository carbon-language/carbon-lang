# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")
load("//testing/fuzzing:rules.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    data = glob(["testdata/**/*.carbon"]),
)

cc_library(
    name = "context",
    srcs = [
        "context.cpp",
        "convert.cpp",
        "decl_name_stack.cpp",
        "deduce.cpp",
        "eval.cpp",
        "function.cpp",
        "generic.cpp",
        "global_init.cpp",
        "import.cpp",
        "import_ref.cpp",
        "inst_block_stack.cpp",
        "merge.cpp",
        "modifiers.cpp",
        "name_component.cpp",
        "return.cpp",
        "subst.cpp",
    ],
    hdrs = [
        "context.h",
        "convert.h",
        "decl_introducer_state.h",
        "decl_name_stack.h",
        "deduce.h",
        "diagnostic_helpers.h",
        "eval.h",
        "function.h",
        "generic.h",
        "global_init.h",
        "import.h",
        "import_ref.h",
        "inst_block_stack.h",
        "keyword_modifier_set.h",
        "merge.h",
        "modifiers.h",
        "name_component.h",
        "param_and_arg_refs_stack.h",
        "pending_block.h",
        "return.h",
        "subst.h",
    ],
    deps = [
        ":node_stack",
        "//common:array_stack",
        "//common:check",
        "//common:map",
        "//common:vlog",
        "//toolchain/base:index_base",
        "//toolchain/base:kind_switch",
        "//toolchain/check:generic_region_stack",
        "//toolchain/check:scope_stack",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/lex:token_kind",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse:node_kind",
        "//toolchain/parse:tree",
        "//toolchain/parse:tree_node_diagnostic_converter",
        "//toolchain/sem_ir:builtin_inst_kind",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:formatter",
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
        ":interface",
        ":member_access",
        ":operator",
        ":pointer_dereference",
        ":sem_ir_diagnostic_converter",
        "//common:check",
        "//common:error",
        "//common:map",
        "//common:ostream",
        "//common:variant_helpers",
        "//common:vlog",
        "//toolchain/base:kind_switch",
        "//toolchain/base:pretty_stack_trace_function",
        "//toolchain/base:value_store",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/lex:token_index",
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
        "//common:exe_path",
        "//testing/fuzzing:libfuzzer_header",
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
        "//toolchain/base:kind_switch",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:ids",
        "//toolchain/sem_ir:inst",
        "//toolchain/sem_ir:inst_kind",
    ],
)

cc_library(
    name = "generic_region_stack",
    srcs = ["generic_region_stack.cpp"],
    hdrs = ["generic_region_stack.h"],
    deps = [
        "//common:array_stack",
        "//common:check",
        "//common:ostream",
        "//common:vlog",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:ids",
        "//toolchain/sem_ir:inst",
        "@llvm-project//llvm:Support",
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
    name = "node_stack",
    srcs = ["node_stack.cpp"],
    hdrs = ["node_stack.h"],
    deps = [
        "//common:check",
        "//common:ostream",
        "//common:vlog",
        "//toolchain/parse:node_kind",
        "//toolchain/parse:tree",
        "//toolchain/sem_ir:formatter",
        "//toolchain/sem_ir:ids",
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

cc_library(
    name = "scope_stack",
    srcs = ["scope_stack.cpp"],
    hdrs = [
        "lexical_lookup.h",
        "scope_index.h",
        "scope_stack.h",
    ],
    deps = [
        "//common:array_stack",
        "//common:check",
        "//common:ostream",
        "//common:set",
        "//common:vlog",
        "//toolchain/base:index_base",
        "//toolchain/sem_ir:file",
        "//toolchain/sem_ir:ids",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "sem_ir_diagnostic_converter",
    srcs = ["sem_ir_diagnostic_converter.cpp"],
    hdrs = ["sem_ir_diagnostic_converter.h"],
    deps = [
        ":context",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/parse:tree_node_diagnostic_converter",
        "//toolchain/sem_ir:file",
        "@llvm-project//llvm:Support",
    ],
)
