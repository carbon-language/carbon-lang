# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:private"])

cc_library(
    name = "ast_to_proto_lib",
    srcs = ["ast_to_proto.cpp"],
    hdrs = ["ast_to_proto.h"],
    deps = [
        "//common/fuzzing:carbon_cc_proto",
        "//executable_semantics/ast",
        "//executable_semantics/ast:declaration",
        "//executable_semantics/ast:expression",
        "//executable_semantics/ast:generic_binding",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "ast_to_proto_test",
    srcs = ["ast_to_proto_test.cpp"],
    args = [
        "$(location //executable_semantics:data/prelude.carbon)",
        "$(locations //executable_semantics/testdata:carbon_files)",
    ],
    data = [
        "//executable_semantics:data/prelude.carbon",
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
