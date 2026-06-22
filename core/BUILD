# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/carbon_rules:defs.bzl", "carbon_library")
load("//bazel/manifest:defs.bzl", "manifest")

# Raw prelude files.
filegroup(
    name = "prelude_files",
    srcs = ["prelude.carbon"] + glob(["prelude/**/*.carbon"]),
    visibility = ["//visibility:public"],
)

# A list of prelude inputs.
# This is consumed by //toolchain/base:install_paths.
manifest(
    name = "prelude_manifest.txt",
    srcs = ["prelude_files"],
    strip_package_dir = True,
)

# All files for the toolchain install.
filegroup(
    name = "prelude",
    srcs = [
        ":prelude_files",
        ":prelude_manifest.txt",
    ],
    visibility = ["//visibility:public"],
)

carbon_library(
    name = "io",
    api = "io.carbon",
    impls = ["io.impl.carbon"],
    visibility = ["//visibility:public"],
)

carbon_library(
    name = "range",
    api = "range.carbon",
    visibility = ["//visibility:public"],
)
