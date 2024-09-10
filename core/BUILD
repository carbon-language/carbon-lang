# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("manifest.bzl", "manifest")

# Raw prelude files.
filegroup(
    name = "prelude_files",
    srcs = ["prelude.carbon"] + glob(["prelude/**/*.carbon"]),
)

# A list of prelude inputs.
# This is consumed by //toolchain/install:install_paths.
manifest(
    name = "prelude_manifest",
    srcs = [":prelude_files"],
    out = "prelude_manifest.txt",
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
