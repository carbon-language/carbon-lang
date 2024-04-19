# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "builtin_kind",
    srcs = ["builtin_kind.cpp"],
    hdrs = ["builtin_kind.h"],
    textual_hdrs = ["builtin_kind.def"],
    deps = ["//common:enum_base"],
)

cc_library(
    name = "block_value_store",
    hdrs = ["block_value_store.h"],
    deps = [
        "//toolchain/base:value_store",
        "//toolchain/base:yaml",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ids",
    hdrs = [
        "id_kind.h",
        "ids.h",
    ],
    deps = [
        "//common:check",
        "//common:ostream",
        "//toolchain/base:index_base",
        "//toolchain/base:value_store",
        "//toolchain/diagnostics:diagnostic_emitter",
        "//toolchain/parse:node_kind",
        "//toolchain/sem_ir:builtin_kind",
    ],
)

cc_library(
    name = "inst_kind",
    srcs = ["inst_kind.cpp"],
    hdrs = [
        "inst_kind.h",
        "typed_insts.h",
    ],
    textual_hdrs = ["inst_kind.def"],
    deps = [
        "//common:enum_base",
        "//toolchain/parse:node_kind",
        "//toolchain/sem_ir:builtin_kind",
        "//toolchain/sem_ir:ids",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "inst",
    srcs = ["inst.cpp"],
    hdrs = ["inst.h"],
    deps = [
        ":block_value_store",
        ":builtin_kind",
        ":ids",
        ":inst_kind",
        "//common:check",
        "//common:ostream",
        "//common:struct_reflection",
        "//toolchain/base:index_base",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "file",
    srcs = [
        "builtin_function_kind.cpp",
        "constant.cpp",
        "file.cpp",
        "inst_profile.cpp",
        "inst_profile.h",
        "name.cpp",
    ],
    hdrs = [
        "builtin_function_kind.h",
        "class.h",
        "constant.h",
        "copy_on_write_block.h",
        "file.h",
        "function.h",
        "impl.h",
        "import_ir.h",
        "interface.h",
        "name.h",
        "name_scope.h",
        "type.h",
    ],
    textual_hdrs = [
        "builtin_function_kind.def",
    ],
    deps = [
        ":block_value_store",
        ":builtin_kind",
        ":ids",
        ":inst",
        ":inst_kind",
        ":type_info",
        "//common:check",
        "//common:enum_base",
        "//common:error",
        "//toolchain/base:kind_switch",
        "//toolchain/base:value_store",
        "//toolchain/base:yaml",
        "//toolchain/lex:token_kind",
        "//toolchain/parse:node_kind",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "inst_namer",
    srcs = ["inst_namer.cpp"],
    hdrs = ["inst_namer.h"],
    deps = [
        ":file",
        ":ids",
        ":inst_kind",
        "//common:ostream",
        "//toolchain/base:kind_switch",
        "//toolchain/base:value_store",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse:tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "formatter",
    srcs = ["formatter.cpp"],
    hdrs = ["formatter.h"],
    deps = [
        ":file",
        ":ids",
        ":inst_kind",
        ":inst_namer",
        "//common:ostream",
        "//toolchain/base:kind_switch",
        "//toolchain/base:value_store",
        "//toolchain/lex:tokenized_buffer",
        "//toolchain/parse:tree",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "entry_point",
    srcs = ["entry_point.cpp"],
    hdrs = ["entry_point.h"],
    deps = [
        ":file",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "type_info",
    hdrs = ["type_info.h"],
    deps = [
        ":ids",
        ":inst",
        "//common:ostream",
    ],
)

cc_test(
    name = "typed_insts_test",
    size = "small",
    srcs = ["typed_insts_test.cpp"],
    deps = [
        ":inst",
        ":inst_kind",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "yaml_test",
    size = "small",
    srcs = ["yaml_test.cpp"],
    deps = [
        "//common:ostream",
        "//testing/base:gtest_main",
        "//testing/base:test_raw_ostream",
        "//toolchain/driver",
        "//toolchain/testing:yaml_test_helpers",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
