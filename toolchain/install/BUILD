# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@llvm-project//llvm:binary_alias.bzl", "binary_alias")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_filegroup", "pkg_files", "pkg_mklink", "strip_prefix")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_pkg//pkg:zip.bzl", "pkg_zip")
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

symlink_filegroup(
    name = "core_data",
    srcs = ["//core:prelude"],
    out_prefix = "prefix_root/lib/carbon/",
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
        ":core_data",
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
        "//common:check",
        "//common:error",
        "@bazel_tools//tools/cpp/runfiles",
        "@llvm-project//llvm:Support",
    ],
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

# Build rules to construct packaged versions of the toolchain's install.

pkg_files(
    name = "packaging_exe_files",
    # We break out the driver and LLD here because we can't easily use file
    # groups that contain symlinks, so those are manually handled below. Other
    # file groups should be directly included.
    srcs = [
        "prefix_root/bin/carbon",
        "prefix_root/lib/carbon/llvm/bin/lld",
    ],
    attributes = pkg_attributes(mode = "0755"),
    strip_prefix = strip_prefix.from_pkg("prefix_root"),
)

# Currently, `rules_pkg` can't replicate symlinks from the main tree. To an
# extent, this is reasonable because we want to be much more explicit about the
# symlink structure in the package where as for the filegroups we're comfortable
# with whatever "just works" for development and testing.
[
    pkg_mklink(
        name = "packaging_link_lld_alias_" + bin_name,
        link_name = "lib/carbon/llvm/bin/" + bin_name,
        target = "lld",
    )
    for bin_name in lld_bin_names
]

pkg_files(
    name = "packaging_core_files",
    srcs = [":core_data"],
    strip_prefix = strip_prefix.from_pkg("prefix_root"),
)

pkg_filegroup(
    name = "packaging_files",
    srcs = [
        ":packaging_core_files",
        ":packaging_exe_files",
    ] + [
        ":packaging_link_lld_alias_" + bin_name
        for bin_name in lld_bin_names
    ],
)

# TODO: We should add support for injecting a version string into both the
# output filename and the package directory name.
pkg_tar(
    name = "carbon_toolchain.tar.bz2.rule",
    srcs = [":packaging_files"],
    out = "carbon_toolchain.tar.bz2",
    extension = "tar.bz2",
    package_dir = "carbon_toolchain",
    tags = ["manual"],  # Slow, exclude from wildcard builds.
)

# TODO: We should add support for injecting a version string into both the
# output filename and the package directory name.
pkg_zip(
    name = "carbon_toolchain.zip.rule",
    srcs = [":packaging_files"],
    out = "carbon_toolchain.zip",
    package_dir = "carbon_toolchain",
    tags = ["manual"],  # Slow, exclude from wildcard builds.
)
