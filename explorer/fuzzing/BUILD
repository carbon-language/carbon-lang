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
        "//explorer/ast",
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
        "//common:error",
        "//common/fuzzing:carbon_cc_proto",
        "//common/fuzzing:proto_to_carbon_lib",
        "//explorer/interpreter:exec_program",
        "//explorer/syntax",
        "//explorer/syntax:prelude",
        "@bazel_tools//tools/cpp/runfiles",
        "@com_google_protobuf//:protobuf_headers",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "ast_to_proto_test",
    srcs = ["ast_to_proto_test.cpp"],
    args = [
        "$(locations //explorer:standard_libraries)",
        "$(locations //explorer/testdata:carbon_files)",
    ],
    data = [
        "//explorer:standard_libraries",
        "//explorer/testdata:carbon_files",
    ],
    deps = [
        ":ast_to_proto_lib",
        "//common/fuzzing:carbon_cc_proto",
        "//explorer/syntax",
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
        "//explorer/common:error_builders",
        "//explorer/common:nonnull",
        "//explorer/syntax",
        "@com_google_protobuf//:protobuf_headers",
        "@llvm-project//llvm:Support",
    ],
)

filegroup(
    name = "fuzzer_corpus_files",
    srcs = glob(["fuzzer_corpus/**"]),
)

cc_test(
    name = "fuzzer_util_test",
    srcs = ["fuzzer_util_test.cpp"],
    args = [
        "$(locations fuzzer_corpus_files)",
    ],
    data = [
        ":fuzzer_corpus_files",
        "//explorer:standard_libraries",
    ],
    deps = [
        ":fuzzer_util",
        "//common:gtest_main",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf_headers",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "proto_to_carbon_test",
    srcs = ["proto_to_carbon_test.cpp"],
    args = [
        "$(locations //explorer:standard_libraries)",
        "$(locations //explorer/testdata:carbon_files)",
    ],
    data = [
        "//explorer:standard_libraries",
        "//explorer/testdata:carbon_files",
    ],
    deps = [
        ":ast_to_proto_lib",
        "//common/fuzzing:carbon_cc_proto",
        "//common/fuzzing:proto_to_carbon_lib",
        "//explorer/syntax",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf_headers",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "explorer_fuzzer",
    testonly = 1,
    srcs = ["explorer_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/*"]),
    deps = [
        ":fuzzer_util",
        "//common:error",
        "@com_google_libprotobuf_mutator//:libprotobuf_mutator",
        "@llvm-project//llvm:Support",
    ],
)
