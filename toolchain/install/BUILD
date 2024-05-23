# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//llvm:binary_alias.bzl", "binary_alias")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("symlink_filegroup.bzl", "symlink_filegroup")

package(default_visibility = ["//visibility:public"])

# Build rules supporting the install data tree for the Carbon toolchain.
#
# The directory `prefix_root` should mirror a Unix-y FHS (Filesystem Hierarchy
# Standard) style install of Carbon under some prefix (`/usr`, `/usr/local`, or
# possibly a home directory or a custom prefix). When installed, user facing
# binaries, which we largely expect to be solely the `carbon` driver will be at
# `perfix_root/bin/carbon`. We collect data, private executables, and other
# internals used by the toolchain in `prefix_root/lib/carbon`.
#
# Code that needs data files in the toolchain should add them to paths below
# `prefix_root/lib/carbon` here and use data dependencies on these paths so that we
# are always twesting against a realistic layout.

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
        "//common:check",
        "//common:command_line",
        "//common:ostream",
        "//common:vlog",
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
)

binary_alias(
    name = "test_direct_exec_root/test_binary",
    testonly = 1,
    binary = ":test_binary",
)

symlink_filegroup(
    name = "test_direct_exec_runfiles",
    testonly = 1,
    srcs = [":install_data"],
    out_prefix = "test_direct_exec_root/test_binary.runfiles/_main/",
)

filegroup(
    name = "test_direct_exec_data",
    testonly = 1,
    srcs = [
        "test_direct_exec_root/test_binary",
        ":test_direct_exec_runfiles",
    ],
)

cc_test(
    name = "install_paths_test",
    size = "small",
    srcs = ["install_paths_test.cpp"],
    data = [
        ":install_data",
        ":test_direct_exec_data",
        ":test_installed_data",
    ],
    deps = [
        ":install_paths",
        "//common:check",
        "//common:ostream",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)
