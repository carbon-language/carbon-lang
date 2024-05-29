# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@llvm-project//llvm:binary_alias.bzl", "binary_alias")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("symlink_filegroup.bzl", "symlink_filegroup")

package(default_visibility = ["//visibility:public"])

# Build rules supporting the install data tree for the Carbon toolchain.
#
# This populates a synthetic Carbon toolchain installation under the
# `prefix_root` directory. For details on its layout, see `install_paths.h`.

# A marker of a valid Carbon install. All filegroups in the install should
# include this one.
write_file(
    name = "install_marker",
    out = "prefix_root/lib/carbon/carbon_install.txt",
    content = [
        "// Part of the Carbon Language project, under the Apache License v2.0 with LLVM",
        "// Exceptions. See /LICENSE for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "This marks a valid Carbon install tree.",
    ],
)

# Copy Clang and LLVM toolchain files into a synthetic LLVM installation under
# `prefix_root/lib/carbon/llvm` so that parts of Clang that expect to find an LLVM
# installation at relative paths work correctly without exposing these in an
# installed 'bin' directory where it might get added to a user's PATH.
binary_alias(
    name = "prefix_root/lib/carbon/llvm/bin/lld",
    binary = "@llvm-project//lld:lld",
)

lld_bin_names = [
    "ld.lld",
    "ld64.lld",
    "lld-link",
    "wasm-ld",
]

[
    binary_alias(
        name = "prefix_root/lib/carbon/llvm/bin/" + bin_name,
        binary = "@llvm-project//lld:lld",
    )
    for bin_name in lld_bin_names
]

filegroup(
    name = "llvm_link_data",
    srcs = [
        "prefix_root/lib/carbon/llvm/bin/lld",
        ":install_marker",
    ] + [
        "prefix_root/lib/carbon/llvm/bin/" + bin_name
        for bin_name in lld_bin_names
    ],
)

# All of the install data except for the user-facing binaries. This is typically
# a reasonable data dependency for libraries and the user-facing binaries
# without creating a cycle.
filegroup(
    name = "install_lib_data",
    srcs = [
        ":install_marker",
        ":llvm_link_data",
    ],
)

binary_alias(
    name = "prefix_root/bin/carbon",
    binary = "//toolchain/driver:carbon",
)

filegroup(
    name = "install_data",
    srcs = [
        "prefix_root/bin/carbon",
        ":install_lib_data",
        ":install_marker",
    ],
)

# A library for computing install paths for the toolchain. Note that this
# library does *not* include the data itself, as that would form a dependency
# cycle. Each part of the toolchain should add the narrow data file groups to
# their data dependencies, and then use this library to locate them.
cc_library(
    name = "install_paths",
    srcs = ["install_paths.cpp"],
    hdrs = ["install_paths.h"],
    deps = [
        "//common:error",
        "@bazel_tools//tools/cpp/runfiles",
        "@llvm-project//llvm:Support",
    ],
)

# Build up some trees of data to use in testing our install detection.
symlink_filegroup(
    name = "test_installed_data",
    testonly = 1,
    srcs = ["//toolchain/install:install_data"],
    out_prefix = "test_installed_root/",
    remove_prefix = "toolchain/install/prefix_root/",
)

cc_binary(
    name = "test_binary",
    testonly = 1,
    srcs = ["test_binary.cpp"],
    data = [":install_data"],
)

cc_test(
    name = "install_paths_test",
    size = "small",
    srcs = ["install_paths_test.cpp"],
    data = [
        ":install_data",
        ":test_binary",
        ":test_installed_data",
    ],
    deps = [
        ":install_paths",
        "//common:check",
        "//common:ostream",
        "//testing/base:gtest_main",
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
