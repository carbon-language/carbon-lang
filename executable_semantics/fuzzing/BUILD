# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/fuzzing:rules.bzl", "cc_fuzz_test")

cc_library(
    name = "ast_to_proto_lib",
    testonly = 1,
    srcs = ["ast_to_proto.cpp"],
    hdrs = ["ast_to_proto.h"],
    deps = [
        "//common/fuzzing:carbon_cc_proto",
        "//executable_semantics/ast",
        "//executable_semantics/ast:declaration",
        "//executable_semantics/ast:expression",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "fuzzer_util",
    testonly = 1,
    srcs = ["fuzzer_util.cpp"],
    hdrs = ["fuzzer_util.h"],
    deps = [
        "//common:check",
        "//common/fuzzing:carbon_cc_proto",
        "//common/fuzzing:proto_to_carbon_lib",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "ast_to_proto_test",
    srcs = ["ast_to_proto_test.cpp"],
    args = [
        "$(locations //executable_semantics:standard_libraries)",
        "$(locations //executable_semantics/testdata:carbon_files)",
    ],
    data = [
        "//executable_semantics:standard_libraries",
        "//executable_semantics/testdata:carbon_files",
    ],
    deps = [
        ":ast_to_proto_lib",
        "//common/fuzzing:carbon_cc_proto",
        "//executable_semantics/syntax",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf_headers",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "fuzzverter",
    testonly = 1,
    srcs = ["fuzzverter.cpp"],
    deps = [
        ":ast_to_proto_lib",
        ":fuzzer_util",
        "//common:error",
        "//common/fuzzing:carbon_cc_proto",
        "//executable_semantics/common:error_builders",
        "//executable_semantics/common:nonnull",
        "//executable_semantics/syntax",
        "@com_google_protobuf//:protobuf_headers",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "proto_to_carbon_test",
    srcs = ["proto_to_carbon_test.cpp"],
    args = [
        "$(locations //executable_semantics:standard_libraries)",
        "$(locations //executable_semantics/testdata:carbon_files)",
    ],
    data = [
        "//executable_semantics:standard_libraries",
        "//executable_semantics/testdata:carbon_files",
    ],
    deps = [
        ":ast_to_proto_lib",
        "//common/fuzzing:carbon_cc_proto",
        "//common/fuzzing:proto_to_carbon_lib",
        "//executable_semantics/syntax",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf_headers",
        "@llvm-project//llvm:Support",
    ],
)

# Needs `--config=proto-fuzzer` for `bazel build` / `bazel test`.
cc_fuzz_test(
    name = "executable_semantics_fuzzer",
    size = "small",
    srcs = ["executable_semantics_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/**"]),
    tags = ["optonly"],
    deps = [
        ":fuzzer_util",
        "//common/fuzzing:carbon_cc_proto",
        "//executable_semantics/interpreter:exec_program",
        "//executable_semantics/syntax",
        "//executable_semantics/syntax:prelude",
        "@com_google_libprotobuf_mutator//:libprotobuf_mutator",
        "@com_google_protobuf//:protobuf_headers",
        "@llvm-project//llvm:Support",
    ],
)
