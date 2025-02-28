# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/manifest:defs.bzl", "manifest")

# Raw prelude files.
# TODO: This includes all of Core, not just the prelude.
filegroup(
    name = "prelude_files",
    srcs = glob(["**/*.carbon"]),
    visibility = ["//visibility:public"],
)

# A list of prelude inputs.
# This is consumed by //toolchain/install:install_paths.
manifest(
    name = "prelude_manifest.txt",
    srcs = [":prelude_files"],
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
