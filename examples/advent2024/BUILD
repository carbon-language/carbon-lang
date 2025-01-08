# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/carbon_rules:defs.bzl", "carbon_binary")

utils = [
    "io_utils.carbon",
    "sort.carbon",
]

# A list of examples that should be excluded because they don't build any more.
# For example, to exclude day3_part2, use:
# excluded = [(3, 2)]
excluded = []

# Produce a binary "dayX_partY" for each matching `.carbon` file.
[
    carbon_binary(
        name = "day{0}_part{1}".format(day, part),
        srcs = [
            "day{0}_common.carbon".format(day),
            "day{0}_part{1}.carbon".format(day, part),
        ] + utils,
    )
    for day in range(1, 14)
    for part in range(1, 3)
    if (day, part) not in excluded
]
