# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "carbon_proto",
    srcs = ["carbon.proto"],
)

cc_proto_library(
    name = "carbon_cc_proto",
    testonly = 1,
    deps = [":carbon_proto"],
)

cc_library(
    name = "proto_to_carbon_lib",
    testonly = 1,
    srcs = ["proto_to_carbon.cpp"],
    hdrs = ["proto_to_carbon.h"],
    deps = [
        ":carbon_cc_proto",
        "@llvm-project//llvm:Support",
    ],
)
