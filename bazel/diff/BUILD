# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_java//java:defs.bzl", "java_binary")

java_binary(
    name = "bazel-diff",
    main_class = "com.bazel_diff.Main",
    runtime_deps = ["@bazel_diff//jar"],
)
