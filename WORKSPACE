# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

workspace(name = "carbon")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Add Bazel's python rules and set up pip.
http_archive(
    name = "rules_python",
    sha256 = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.1.0/rules_python-0.1.0.tar.gz",
)

load("@rules_python//python:pip.bzl", "pip_install")

# Create a central repo that knows about the pip dependencies.
pip_install(
    name = "py_deps",
    requirements = "//github_tools:requirements.txt",
)

# Bootstrap a Clang and LLVM toolchain.
load("//bazel/cc_toolchains:clang_bootstrap.bzl", "bootstrap_clang_toolchain")

bootstrap_clang_toolchain(name = "bootstrap_clang_toolchain")

# Configure the bootstrapped Clang and LLVM toolchain for Bazel.
load("//bazel/cc_toolchains:clang_configuration.bzl", "configure_clang_toolchain")

configure_clang_toolchain(
    name = "bazel_cc_toolchain",
    clang = "@bootstrap_clang_toolchain//:bin/clang",
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

# TODO(chandlerc): Replace this with an upstream release once the pull request
# with our needed functionality lands:
# https://github.com/jmillikin/rules_m4/pull/7
#
# Until then, this is pulling from that pull request's commit.
http_archive(
    name = "rules_m4",
    sha256 = "4d34917214e8890ad770bdf0c319c41c9201fffd770938b41a1d641d4b27e05c",
    strip_prefix = "rules_m4-add-extra-copts",
    urls = ["https://github.com/chandlerc/rules_m4/archive/add-extra-copts.zip"],
)

load("@rules_m4//m4:m4.bzl", "m4_register_toolchains")

# When building M4, disable all compiler warnings as we can't realistically fix
# them anyways.
m4_register_toolchains(extra_copts = ["-w"])

# TODO(chandlerc): Replace this with an upstream release once the pull request
# with our needed functionality lands:
# https://github.com/jmillikin/rules_flex/pull/5
#
# Until then, this is pulling from that pull request's commit.
http_archive(
    name = "rules_flex",
    sha256 = "fd97c3ae23926507be1b95158a683cd41c628d201e852a325d38b5e9f821b752",
    strip_prefix = "rules_flex-add-extra-copts",
    urls = ["https://github.com/chandlerc/rules_flex/archive/add-extra-copts.zip"],
)

load("@rules_flex//flex:flex.bzl", "flex_register_toolchains")

# When building Flex, disable all compiler warnings as we can't realistically
# fix them anyways.
flex_register_toolchains(extra_copts = ["-w"])

# TODO(chandlerc): Replace this with an upstream release once the pull request
# with our needed functionality lands:
# https://github.com/jmillikin/rules_bison/pull/7
#
# Until then, this is pulling from that pull request's commit.
http_archive(
    name = "rules_bison",
    sha256 = "c6e926f15214d903966dc950d759ec69116db67f148be114c119e4def0551eaa",
    strip_prefix = "rules_bison-add-extra-copts",
    urls = ["https://github.com/chandlerc/rules_bison/archive/add-extra-copts.zip"],
)

load("@rules_bison//bison:bison.bzl", "bison_register_toolchains")

# When building Bison, disable all compiler warnings as we can't realistically
# fix them anyways.
bison_register_toolchains(extra_copts = ["-w"])

local_repository(
    name = "brotli",
    path = "third_party/examples/brotli/original",
)

new_local_repository(
    name = "woff2",
    build_file = "third_party/examples/woff2/BUILD.original",
    path = "third_party/examples/woff2/original",
    workspace_file = "third_party/examples/woff2/WORKSPACE.original",
)
