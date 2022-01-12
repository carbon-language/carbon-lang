# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

workspace(name = "carbon")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

###############################################################################
# Python rules
###############################################################################

rules_python_version = "0.5.0"

# Add Bazel's python rules and set up pip.
http_archive(
    name = "rules_python",
    sha256 = "cd6730ed53a002c56ce4e2f396ba3b3be262fd7cb68339f0377a45e8227fe332",
    url = "https://github.com/bazelbuild/rules_python/releases/download/%s/rules_python-%s.tar.gz" % (
        rules_python_version,
        rules_python_version,
    ),
)

load("@rules_python//python:pip.bzl", "pip_install")

# Create a central repo that knows about the pip dependencies.
pip_install(
    name = "py_deps",
    requirements = "//github_tools:requirements.txt",
)

###############################################################################
# Python mypy rules
###############################################################################

# NOTE: https://github.com/bazelbuild/bazel/issues/4948 tracks bazel supporting
# typing directly. If it's added, we will probably want to switch.

# Add mypy
mypy_integration_version = "e5f8071f33eca637cd90bf70cb45f749e63bf2ca"

# TODO: Can switch back to the official repo when it includes:
# https://github.com/thundergolfer/bazel-mypy-integration/pull/43
#http_archive(
#    name = "mypy_integration",
#    sha256 = "621df076709dc72809add1f5fe187b213fee5f9b92e39eb33851ab13487bd67d",
#    strip_prefix = "bazel-mypy-integration-%s" % mypy_integration_version,
#    urls = [
#        "https://github.com/thundergolfer/bazel-mypy-integration/archive/refs/tags/%s.tar.gz" % mypy_integration_version,
#    ],
#)

http_archive(
    name = "mypy_integration",
    sha256 = "481ec6f0953a84a36b8103286f04c4cd274ae689060099085c02ac187d007592",
    strip_prefix = "bazel-mypy-integration-%s" % mypy_integration_version,
    urls = [
        "https://github.com/jonmeow/bazel-mypy-integration/archive/%s.zip" % mypy_integration_version,
    ],
)

load(
    "@mypy_integration//repositories:repositories.bzl",
    mypy_integration_repositories = "repositories",
)

mypy_integration_repositories()

load("@mypy_integration//:config.bzl", "mypy_configuration")

mypy_configuration("//bazel/mypy:mypy.ini")

load("@mypy_integration//repositories:deps.bzl", mypy_integration_deps = "deps")

mypy_integration_deps(
    mypy_requirements_file = "//bazel/mypy:version.txt",
)

###############################################################################
# C++ rules
###############################################################################

# Configure the bootstrapped Clang and LLVM toolchain for Bazel.
load(
    "//bazel/cc_toolchains:clang_configuration.bzl",
    "configure_clang_toolchain",
)

configure_clang_toolchain(name = "bazel_cc_toolchain")

###############################################################################
# Abseil libraries
###############################################################################

abseil_version = "20211102.0"

http_archive(
    name = "com_google_absl",
    sha256 = "a4567ff02faca671b95e31d315bab18b42b6c6f1a60e91c6ea84e5a2142112c2",
    strip_prefix = "abseil-cpp-%s" % abseil_version,
    urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/%s.zip" % abseil_version],
)

###############################################################################
# GoogleTest libraries
###############################################################################

# Version as of 2021-12-07. Not a major release, but gets a clang-tidy fix.
googletest_version = "4c5650f68866e3c2e60361d5c4c95c6f335fb64b"

http_archive(
    name = "com_google_googletest",
    sha256 = "238ee428a2cde2f07c6925e9e2d237dc5aad52532c6ba584cb260d46d7b78455",
    strip_prefix = "googletest-%s" % googletest_version,
    urls = ["https://github.com/google/googletest/archive/%s.zip" % googletest_version],
)

###############################################################################
# Google Benchmark libraries
###############################################################################

benchmark_version = "1.6.0"

http_archive(
    name = "com_github_google_benchmark",
    sha256 = "3da225763533aa179af8438e994842be5ca72e4a7fed4d7976dc66c8c4502f58",
    strip_prefix = "benchmark-%s" % benchmark_version,
    urls = ["https://github.com/google/benchmark/archive/refs/tags/v%s.zip" % benchmark_version],
)

###############################################################################
# LLVM libraries
###############################################################################

llvm_version = "dbf0d8118ccbfd09def0974bf671c39e94d0be93"
http_archive(
    name = "llvm-raw",
    build_file_content = "# empty",
    sha256 = "16b368284aabf4bb94bbd761a009401811d99e3288fb346877072f32926f33db",
    strip_prefix = "llvm-project-%s" % llvm_version,
    urls = ["https://github.com/llvm/llvm-project/archive/%s.zip" % llvm_version],
)

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

llvm_configure(
    name = "llvm-project",
    targets = [
        "AArch64",
        "X86",
    ],
)

load("@llvm-raw//utils/bazel:terminfo.bzl", "llvm_terminfo_system")

# We require successful detection and use of a system terminfo library.
llvm_terminfo_system(name = "llvm_terminfo")

load("@llvm-raw//utils/bazel:zlib.bzl", "llvm_zlib_system")

# We require successful detection and use of a system zlib library.
llvm_zlib_system(name = "llvm_zlib")

###############################################################################
# Flex/Bison rules
###############################################################################

# TODO: Can switch to a normal release version when it includes:
# https://github.com/jmillikin/rules_m4/commit/b504241407916d1d6d72c66a766daacf9603cf8b
rules_m4_version = "b504241407916d1d6d72c66a766daacf9603cf8b"

http_archive(
    name = "rules_m4",
    sha256 = "e6003c5f45746a2ad01335a8526044591f2b6c5c68852cee1bcd28adc2cf452b",
    strip_prefix = "rules_m4-%s" % rules_m4_version,
    urls = ["https://github.com/jmillikin/rules_m4/archive/%s.zip" %
            rules_m4_version],
)

load("@rules_m4//m4:m4.bzl", "m4_register_toolchains")

# When building M4, disable all compiler warnings as we can't realistically fix
# them anyways.
m4_register_toolchains(extra_copts = ["-w"])

# TODO: Can switch to a normal release version when it includes:
# https://github.com/jmillikin/rules_flex/commit/1f1d9c306c2b4b8be2cb899a3364b84302124e77
rules_flex_version = "1f1d9c306c2b4b8be2cb899a3364b84302124e77"

http_archive(
    name = "rules_flex",
    sha256 = "ad1c3a1a9bdd6254df857f84f3ab4c052df6e21ce4af5d32710f2feff2abf4dd",
    strip_prefix = "rules_flex-%s" % rules_flex_version,
    urls = ["https://github.com/jmillikin/rules_flex/archive/%s.zip" %
            rules_flex_version],
)

load("@rules_flex//flex:flex.bzl", "flex_register_toolchains")

# When building Flex, disable all compiler warnings as we can't realistically
# fix them anyways.
flex_register_toolchains(extra_copts = ["-w"])

# TODO: Can switch to a normal release version when it includes:
# https://github.com/jmillikin/rules_bison/commit/478079b28605a38000eaf83719568d756b3383a0
rules_bison_version = "478079b28605a38000eaf83719568d756b3383a0"

http_archive(
    name = "rules_bison",
    sha256 = "d662d200f4e2a868f6873d666402fa4d413f07ba1a433591c5f60ac601157fb9",
    strip_prefix = "rules_bison-%s" % rules_bison_version,
    urls = ["https://github.com/jmillikin/rules_bison/archive/%s.zip" %
            rules_bison_version],
)

load("@rules_bison//bison:bison.bzl", "bison_register_toolchains")

# When building Bison, disable all compiler warnings as we can't realistically
# fix them anyways.
bison_register_toolchains(extra_copts = ["-w"])

###############################################################################
# Example conversion repositories
###############################################################################

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

local_repository(
    name = "woff2_carbon",
    path = "third_party/examples/woff2/carbon",
)
