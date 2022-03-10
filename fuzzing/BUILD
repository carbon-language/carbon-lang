# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "carbon_proto",
    srcs = ["carbon.proto"],
)

cc_proto_library(
    name = "carbon_cc_proto",
    deps = [":carbon_proto"],
)
