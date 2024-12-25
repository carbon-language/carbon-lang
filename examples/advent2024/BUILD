# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/carbon_rules:defs.bzl", "carbon_binary")

utils = [
    "io_utils.carbon",
    "sort.carbon",
]

[
    carbon_binary(
        name = "day%d_part%d" % (day, part),
        srcs = [
            "day%d_common.carbon" % day,
            "day%d_part%d.carbon" % (day, part),
        ] + utils,
    )
    for day in range(1, 11)
    for part in range(1, 3)
]
