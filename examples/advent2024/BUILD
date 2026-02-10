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
# excluded = ["day3_part2.carbon"]
excluded = []

# Produce a binary `dayX_partY` for each matching `.carbon` file. Each binary
# depends on `dayX_partY.carbon` and `dayX_common.carbon`.
[
    carbon_binary(
        name = carbon_file.removesuffix(".carbon"),
        srcs = [
            carbon_file,
            carbon_file.rsplit("_", 1)[0] + "_common.carbon",
        ] + utils,
    )
    for carbon_file in glob(["day*_part*.carbon"])
    if carbon_file not in excluded
]
