# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:private"])

cc_library(
    name = "ast_to_proto_lib",
    srcs = ["ast_to_proto.cpp"],
    hdrs = ["ast_to_proto.h"],
    deps = [
        "//executable_semantics/ast",
        "//executable_semantics/ast:declaration",
        "//executable_semantics/ast:expression",
        "//executable_semantics/ast:generic_binding",
        "//fuzzing:carbon_cc_proto",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "ast_to_proto_test",
    srcs = ["ast_to_proto_test.cpp"],
    copts = [
        "-Wno-thread-safety-reference",
        "-Wno-implicit-fallthrough",
    ],
    deps = [
        ":ast_to_proto_lib",
        "//executable_semantics/syntax",
        "//fuzzing:carbon_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf_headers",
        "@llvm-project//llvm:Support",
    ],
)
