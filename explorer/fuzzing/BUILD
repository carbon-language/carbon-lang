# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//testing/fuzzing:rules.bzl", "cc_fuzz_test")

cc_library(
    name = "ast_to_proto_lib",
    testonly = 1,
    srcs = ["ast_to_proto.cpp"],
    hdrs = ["ast_to_proto.h"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        "//explorer/ast",
        "//testing/fuzzing:carbon_cc_proto",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "ast_to_proto",
    testonly = 1,
    srcs = ["ast_to_proto_main.cpp"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":ast_to_proto_lib",
        "//common:bazel_working_dir",
        "//common:error",
        "//explorer/ast",
        "//explorer/base:arena",
        "//explorer/syntax",
        "//testing/fuzzing:carbon_cc_proto",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

cc_test(
    name = "ast_to_proto_test",
    size = "small",
    srcs = ["ast_to_proto_test.cpp"],
    args = [
        "$(locations //explorer:standard_libraries)",
        "$(locations //explorer:carbon_files)",
    ],
    data = [
        "//explorer:carbon_files",
        "//explorer:standard_libraries",
    ],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":ast_to_proto_lib",
        "//explorer/syntax",
        "//testing/base:test_raw_ostream",
        "//testing/fuzzing:carbon_cc_proto",
        "//testing/fuzzing:proto_to_carbon_lib",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

cc_library(
    name = "fuzzer_util",
    testonly = 1,
    srcs = ["fuzzer_util.cpp"],
    hdrs = ["fuzzer_util.h"],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        "//common:check",
        "//common:error",
        "//explorer/ast",
        "//explorer/parse_and_execute",
        "//testing/fuzzing:carbon_cc_proto",
        "//testing/fuzzing:proto_to_carbon_lib",
        "@bazel_tools//tools/cpp/runfiles",
        "@com_google_protobuf//:protobuf_headers",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "fuzzer_util_test",
    size = "small",
    srcs = ["fuzzer_util_test.cpp"],
    data = [
        "//explorer:standard_libraries",
    ],
    # Running clang-tidy is slow, and explorer is currently feature frozen, so
    # don't spend time linting it.
    tags = ["no-clang-tidy"],
    deps = [
        ":fuzzer_util",
        "//testing/base:gtest_main",
        "//testing/fuzzing:proto_to_carbon_lib",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf_headers",
        "@llvm-project//llvm:Support",
    ],
)

cc_fuzz_test(
    name = "explorer_fuzzer",
    size = "small",
    srcs = ["explorer_fuzzer.cpp"],
    corpus = glob(["fuzzer_corpus/*"]),
    shard_count = 8,
    tags = [
        "no-clang-tidy",
        "proto-fuzzer",
    ],
    deps = [
        ":fuzzer_util",
        "//common:error",
        "@com_google_libprotobuf_mutator//:libprotobuf_mutator",
        "@llvm-project//llvm:Support",
    ],
)
