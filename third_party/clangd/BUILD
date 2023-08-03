# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "clangd",
    srcs = [
        "Protocol.cpp",
        "Transport.cpp",
        "URI.cpp",
        "support/Logger.cpp",
    ],
    hdrs = [
        "Protocol.h",
        "Transport.h",
        "URI.h",
        "support/Function.h",
        "support/Logger.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)
