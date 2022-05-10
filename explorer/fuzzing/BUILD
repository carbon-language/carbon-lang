# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cc_library(
    name = "ast_to_proto_lib",
    srcs = ["ast_to_proto.cpp"],
    hdrs = ["ast_to_proto.h"],
    deps = [
        "//common/fuzzing:carbon_cc_proto",
        "//explorer/ast",
        "//explorer/ast:declaration",
        "//explorer/ast:expression",
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
