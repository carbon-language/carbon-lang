# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

workspace(name = "carbon")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# We want to use LLVM via an external CMake build, so pull in the Bazel
# infrastructure that provides direct CMake interfacing support.
http_archive(
    name = "rules_foreign_cc",
    strip_prefix = "rules_foreign_cc-master",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/master.zip",
)

# Add Bazel's python rules.
http_archive(
    name = "rules_python",
    sha256 = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.1.0/rules_python-0.1.0.tar.gz",
)

# Set up necessary dependencies for working with the foreign C++ rules.
load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

# Bootstrap a Clang toolchain.
load("//bazel/cc_toolchains:clang_bootstrap.bzl", "bootstrap_clang_toolchain")

bootstrap_clang_toolchain(name = "bootstrap_clang_toolchain")

# Detect and configure a Clang and LLVM based toolchain.
load("//bazel/cc_toolchains:clang_detection.bzl", "configure_clang_toolchain")

configure_clang_toolchain(
    name = "bazel_cc_toolchain",
    clang = "@bootstrap_clang_toolchain//bin/clang",
)

local_repository(
    name = "llvm_bazel",
    path = "third_party/llvm-bazel/llvm-bazel",
)

load("@llvm_bazel//:configure.bzl", "llvm_configure")

llvm_configure(
    name = "llvm-project",
    src_path = "third_party/llvm-project",
    src_workspace = "@carbon//:WORKSPACE",
)

load("@llvm_bazel//:terminfo.bzl", "llvm_terminfo_system")

# We require successful detection and use of a system terminfo library.
llvm_terminfo_system(name = "llvm_terminfo")

load("@llvm_bazel//:zlib.bzl", "llvm_zlib_system")

# We require successful detection and use of a system zlib library.
llvm_zlib_system(name = "llvm_zlib")
