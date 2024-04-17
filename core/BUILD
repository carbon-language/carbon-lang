# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

filegroup(
    name = "core",
    data = glob(["**/*.carbon"]),
    visibility = ["//visibility:public"],
)

# The prelude itself. This is a dependency of compilations but not an input.
filegroup(
    name = "prelude",
    srcs = ["prelude.carbon"],
    visibility = ["//visibility:public"],
)

# The dependencies of the prelude.
filegroup(
    name = "prelude_deps",
    srcs = glob(["prelude/**/*.carbon"]),
    visibility = ["//visibility:public"],
)
