# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

workspace(name = "carbon")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

skylib_version = "1.3.0"

http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/{0}/bazel-skylib-{0}.tar.gz".format(skylib_version),
        "https://github.com/bazelbuild/bazel-skylib/releases/download/{0}/bazel-skylib-{0}.tar.gz".format(skylib_version),
    ],
)

###############################################################################
# Python rules
###############################################################################

rules_python_version = "0.8.1"

# Add Bazel's python rules and set up pip.
http_archive(
    name = "rules_python",
    sha256 = "cdf6b84084aad8f10bf20b46b77cb48d83c319ebe6458a18e9d2cebf57807cdd",
    strip_prefix = "rules_python-{0}".format(rules_python_version),
    url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/{0}.tar.gz".format(rules_python_version),
)

load("@rules_python//python:pip.bzl", "pip_install")

# Create a central repo that knows about the pip dependencies.
pip_install(
    name = "py_deps",
    requirements = "//github_tools:requirements.txt",
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

# Head as of 2022-09-13.
abseil_version = "530cd52f585c9d31b2b28cea7e53915af7a878e3"

http_archive(
    name = "com_google_absl",
    sha256 = "f8a6789514a3b109111252af92da41d6e64f90efca9fb70515d86debee57dc24",
    strip_prefix = "abseil-cpp-{0}".format(abseil_version),
    urls = ["https://github.com/abseil/abseil-cpp/archive/{0}.tar.gz".format(abseil_version)],
)

###############################################################################
# RE2 libraries
###############################################################################

# Head as of 2022-09-14.
re2_version = "cc1c9db8bf5155d89d10d65998cdb226f676492c"

http_archive(
    name = "com_googlesource_code_re2",
    sha256 = "8ef976c79a300f8c5e880535665bd4ba146fb09fb6d2342f8f1a02d9af29f365",
    strip_prefix = "re2-{0}".format(re2_version),
    urls = ["https://github.com/google/re2/archive/{0}.tar.gz".format(re2_version)],
)

###############################################################################
# GoogleTest libraries
###############################################################################

# Head as of 2022-09-14.
googletest_version = "1336c4b6d1a6f4bc6beebccb920e5ff858889292"

http_archive(
    name = "com_google_googletest",
    sha256 = "d701aaeb9a258afba27210d746d971042be96c371ddc5a49f1e8914d9ea17e3c",
    strip_prefix = "googletest-{0}".format(googletest_version),
    urls = ["https://github.com/google/googletest/archive/{0}.tar.gz".format(googletest_version)],
)

###############################################################################
# Google Benchmark libraries
###############################################################################

benchmark_version = "1.6.1"

http_archive(
    name = "com_github_google_benchmark",
    sha256 = "6132883bc8c9b0df5375b16ab520fac1a85dc9e4cf5be59480448ece74b278d4",
    strip_prefix = "benchmark-{0}".format(benchmark_version),
    urls = ["https://github.com/google/benchmark/archive/refs/tags/v{0}.tar.gz".format(benchmark_version)],
)

###############################################################################
# LLVM libraries
###############################################################################

# We pin to specific upstream commits and try to track top-of-tree reasonably
# closely rather than pinning to a specific release.
llvm_version = "ecfa2d3d9943a48411d04a4b3103c42b4653d9af"

http_archive(
    name = "llvm-raw",
    build_file_content = "# empty",
    patch_args = ["-p1"],
    patches = [
        "@carbon//bazel/patches/llvm:0001_Patch_for_mallinfo2_when_using_Bazel_build_system.patch",
        "@carbon//bazel/patches/llvm:0002_Added_Bazel_build_for_compiler_rt_fuzzer.patch",
    ],
    sha256 = "8e9cbb937b1a40536cd809e09603a1810d86a8c314fee0cca36fc493e78289e5",
    strip_prefix = "llvm-project-{0}".format(llvm_version),
    urls = ["https://github.com/llvm/llvm-project/archive/{0}.tar.gz".format(llvm_version)],
)

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

llvm_configure(
    name = "llvm-project",
    repo_mapping = {"@llvm_zlib": "@zlib"},
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
llvm_zlib_system(name = "zlib")

###############################################################################
# Flex/Bison rules
###############################################################################

rules_m4_version = "0.2.1"

http_archive(
    name = "rules_m4",
    patch_args = ["-p1"],
    patches = [
        # Trying to upstream: https://github.com/jmillikin/rules_m4/pull/15
        "@carbon//bazel/patches/m4:0001_Support_M4_building_on_FreeBSD.patch",
    ],
    sha256 = "eaa674cd84546038ecbcc49cdd346134a20961a41fa1a541e80d8bf4b470c34d",
    strip_prefix = "rules_m4-{0}".format(rules_m4_version),
    urls = ["https://github.com/jmillikin/rules_m4/archive/v{0}.tar.gz".format(rules_m4_version)],
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
    sha256 = "a4e99a0a241c8a5aa238e81724ea3529722522c3702fd3aa674add5eb9807002",
    strip_prefix = "rules_flex-{0}".format(rules_flex_version),
    urls = ["https://github.com/jmillikin/rules_flex/archive/{0}.tar.gz".format(rules_flex_version)],
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
    patch_args = ["-p1"],
    patches = [
        # Trying to upstream: https://github.com/jmillikin/rules_bison/pull/13
        "@carbon//bazel/patches/bison:0001_Support_Bison_building_on_FreeBSD.patch",
    ],
    sha256 = "6bc2d382e4ffccd66e60a74521c24722fc8fdfe9af49ff182f79bb5994fa1ba4",
    strip_prefix = "rules_bison-{0}".format(rules_bison_version),
    urls = ["https://github.com/jmillikin/rules_bison/archive/{0}.tar.gz".format(rules_bison_version)],
)

load("@rules_bison//bison:bison.bzl", "bison_register_toolchains")

# When building Bison, disable all compiler warnings as we can't realistically
# fix them anyways.
bison_register_toolchains(extra_copts = ["-w"])

###############################################################################
# Protocol buffers - for structured fuzzer testing.
###############################################################################

rules_cc_version = "0.0.4"

http_archive(
    name = "rules_cc",
    sha256 = "af6cc82d87db94585bceeda2561cb8a9d55ad435318ccb4ddfee18a43580fb5d",
    strip_prefix = "rules_cc-{0}".format(rules_cc_version),
    urls = ["https://github.com/bazelbuild/rules_cc/releases/download/{0}/rules_cc-{0}.tar.gz".format(rules_cc_version)],
)

rules_proto_version = "5.3.0-21.7"

http_archive(
    name = "rules_proto",
    sha256 = "dc3fb206a2cb3441b485eb1e423165b231235a1ea9b031b4433cf7bc1fa460dd",
    strip_prefix = "rules_proto-{0}".format(rules_proto_version),
    urls = ["https://github.com/bazelbuild/rules_proto/archive/refs/tags/{0}.tar.gz".format(rules_proto_version)],
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()

###############################################################################
# libprotobuf_mutator - for structured fuzzer testing.
###############################################################################

libprotobuf_mutator_version = "1.1"

http_archive(
    name = "com_google_libprotobuf_mutator",
    build_file = "@//:third_party/libprotobuf_mutator/BUILD.txt",
    sha256 = "fd299fd72c5cf664259d9bd43a72cb74dc6a8b9604d107fe2d2e90885aeb7c16",
    strip_prefix = "libprotobuf-mutator-{0}".format(libprotobuf_mutator_version),
    urls = ["https://github.com/google/libprotobuf-mutator/archive/v{0}.tar.gz".format(libprotobuf_mutator_version)],
)

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
