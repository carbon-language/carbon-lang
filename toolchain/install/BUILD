# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//bazel/cc_toolchains:defs.bzl", "cc_env", "llvm_symbolizer")
load("install_filegroups.bzl", "install_filegroup", "install_symlink", "install_target", "make_install_filegroups")
load("pkg_helpers.bzl", "pkg_naming_variables", "pkg_tar_and_test")
load("run_tool.bzl", "run_tool")

package(default_visibility = ["//visibility:public"])

# Build rules supporting the install data tree for the Carbon toolchain.
#
# This populates a synthetic Carbon toolchain installation under the
# `prefix_root` directory. For details on its layout, see `install_paths.h`.

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
        "//testing/base:global_exe_path",
        "//testing/base:gtest_main",
        "@bazel_tools//tools/cpp/runfiles",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "install_paths_test_helpers",
    testonly = 1,
    srcs = ["install_paths_test_helpers.cpp"],
    hdrs = ["install_paths_test_helpers.h"],
    deps = [
        ":install_paths",
        "//testing/base:global_exe_path",
        "@llvm-project//llvm:Support",
    ],
)

lld_aliases = [
    "ld.lld",
    "ld64.lld",
    "lld-link",
    "wasm-ld",
]

install_dirs = {
    "bin": [
        install_target(
            "carbon",
            "//toolchain/driver:carbon",
            executable = True,
            is_driver = True,
        ),
    ],
    "lib/carbon": [
        install_target("carbon_install.txt", "carbon_install.txt"),
        install_filegroup("core", "//core:prelude"),
    ],
    "lib/carbon/llvm/bin": [
        install_target(
            "lld",
            "@llvm-project//lld:lld",
            executable = True,
        ),
    ] + [install_symlink(name, "lld") for name in lld_aliases],
}

make_install_filegroups(
    name = "install_data",
    install_dirs = install_dirs,
    no_driver_name = "install_data.no_driver",
    pkg_name = "pkg_data",
    prefix = "prefix_root",
)

pkg_naming_variables(
    name = "packaging_variables",
)

# We build both a compressed and uncompressed tar file with the same code here.
# This lets us use the tar file in testing as it is fast to create, but ship the
# compressed version as a release.
pkg_tar_and_test(
    srcs = [":pkg_data"],
    name_base = "carbon_toolchain",
    package_dir = "carbon_toolchain-$(version)",
    package_file_name_base = "carbon_toolchain-$(version)",
    package_variables = ":packaging_variables",
    stamp = -1,  # Allow `--stamp` builds to produce file timestamps.
    test_data = [
        ":install_data",
    ],
    # TODO: This is used to make sure that tar files are in install_data (one
    # direction). Replace with a check that the files in install_data and tar
    # match (bidirectional).
    test_install_marker = "prefix_root/lib/carbon/carbon_install.txt",
)

# Support `bazel run` on specific binaries.
run_tool(
    name = "run_carbon",
    data = [
        ":install_data",
        llvm_symbolizer,
    ],
    env = cc_env(),
    tool = "prefix_root/bin/carbon",
)
