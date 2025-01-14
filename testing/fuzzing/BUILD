# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Header for LibFuzzer, does not provide the implementation which should come
# from some other source such as a fuzz test target.
cc_library(
    name = "libfuzzer_header",
    testonly = 1,
    hdrs = ["libfuzzer.h"],
)
