# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Workspace rules supporting C++."""

load("//bazel/cc_toolchains:clang_bootstrap.bzl", "bootstrap_clang_toolchain")
load("//bazel/cc_toolchains:clang_configuration.bzl", "configure_clang_toolchain")
load("@llvm_bazel//:configure.bzl", "llvm_configure")
load("@llvm_bazel//:terminfo.bzl", "llvm_terminfo_system")
load("@llvm_bazel//:zlib.bzl", "llvm_zlib_system")

def cc_workspace_load():
    # Bootstrap a Clang and LLVM toolchain.
    bootstrap_clang_toolchain(name = "bootstrap_clang_toolchain")

    # Configure the bootstrapped Clang and LLVM toolchain for Bazel.
    configure_clang_toolchain(
        name = "bazel_cc_toolchain",
        clang = "@bootstrap_clang_toolchain//:bin/clang",
    )

    llvm_configure(
        name = "llvm-project",
        src_path = "third_party/llvm-project",
        src_workspace = "@carbon//:WORKSPACE",
    )

    # We require successful detection and use of a system terminfo library.
    llvm_terminfo_system(name = "llvm_terminfo")

    # We require successful detection and use of a system zlib library.
    llvm_zlib_system(name = "llvm_zlib")
