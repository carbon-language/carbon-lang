# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

workspace(name = "carbon")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

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

# Head as of 2023-07-31.
abseil_version = "407f2fdd5ec6f79287919486aa5869b346093906"

http_archive(
    name = "com_google_absl",
    sha256 = "953a914ac42f87caf5ed6a86890e183ae4e2bc69666a90c67605091d6e77e502",
    strip_prefix = "abseil-cpp-{0}".format(abseil_version),
    urls = ["https://github.com/abseil/abseil-cpp/archive/{0}.tar.gz".format(abseil_version)],
)

###############################################################################
# RE2 libraries
###############################################################################

# Head as of 2023-07-31.
re2_version = "960c861764ff54c9a12ff683ba55ccaad1a8f73b"

http_archive(
    name = "com_googlesource_code_re2",
    sha256 = "8315f22198c25e9f7f1a3754566824710c08ddbb39d93e9920f4a131e871fc15",
    strip_prefix = "re2-{0}".format(re2_version),
    urls = ["https://github.com/google/re2/archive/{0}.tar.gz".format(re2_version)],
)

###############################################################################
# GoogleTest libraries
###############################################################################

# Head as of 2023-07-31.
googletest_version = "c875c4e2249ec124c24f72141b3780c22256fd44"

http_archive(
    name = "com_google_googletest",
    sha256 = "21e0cd1110ba534409facccdda1bad90174e7ee7ded60c00dd2b43b4df654080",
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
llvm_version = "3d51010a3350660160981c6b8e624dcc87c208a3"

http_archive(
    name = "llvm-raw",
    build_file_content = "# empty",
    patch_args = ["-p1"],
    patches = [
        "@carbon//bazel/patches/llvm:0001_Patch_for_mallinfo2_when_using_Bazel_build_system.patch",
        "@carbon//bazel/patches/llvm:0002_Added_Bazel_build_for_compiler_rt_fuzzer.patch",
        "@carbon//bazel/patches/llvm:0003_Add_library_for_clangd.patch",
    ],
    sha256 = "efbca707a6eb1c714b849de120309070eef282660c0f4be5b68efef62cc95cf5",
    strip_prefix = "llvm-project-{0}".format(llvm_version),
    urls = ["https://github.com/llvm/llvm-project/archive/{0}.tar.gz".format(llvm_version)],
)

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

llvm_configure(
    name = "llvm-project",
    targets = [
        "AArch64",
        "X86",
    ],
)

# Dependencies copied from
# https://github.com/llvm/llvm-project/blob/main/utils/bazel/WORKSPACE.
maybe(
    http_archive,
    name = "llvm_zlib",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
    sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
    strip_prefix = "zlib-ng-2.0.7",
    urls = [
        "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
    ],
)

maybe(
    http_archive,
    name = "llvm_zstd",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
    sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
    strip_prefix = "zstd-1.5.2",
    urls = [
        "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
    ],
)

###############################################################################
# Flex/Bison rules
###############################################################################

rules_m4_version = "0.2.3"

http_archive(
    name = "rules_m4",
    sha256 = "10ce41f150ccfbfddc9d2394ee680eb984dc8a3dfea613afd013cfb22ea7445c",
    urls = ["https://github.com/jmillikin/rules_m4/releases/download/v{0}/rules_m4-v{0}.tar.xz".format(rules_m4_version)],
)

load("@rules_m4//m4:m4.bzl", "m4_register_toolchains")

# When building M4, disable all compiler warnings as we can't realistically fix
# them anyways.
m4_register_toolchains(extra_copts = ["-w"])

rules_flex_version = "0.2.1"

http_archive(
    name = "rules_flex",
    sha256 = "8929fedc40909d19a4b42548d0785f796c7677dcef8b5d1600b415e5a4a7749f",
    urls = ["https://github.com/jmillikin/rules_flex/releases/download/v{0}/rules_flex-v{0}.tar.xz".format(rules_flex_version)],
)

load("@rules_flex//flex:flex.bzl", "flex_register_toolchains")

# When building Flex, disable all compiler warnings as we can't realistically
# fix them anyways.
flex_register_toolchains(extra_copts = ["-w"])

rules_bison_version = "0.2.2"

http_archive(
    name = "rules_bison",
    sha256 = "2279183430e438b2dc77cacd7b1dbb63438971b2411406570f1ddd920b7c9145",
    urls = ["https://github.com/jmillikin/rules_bison/releases/download/v{0}/rules_bison-v{0}.tar.xz".format(rules_bison_version)],
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

###############################################################################
# Treesitter rules
###############################################################################

http_archive(
    name = "rules_nodejs",
    sha256 = "d124665ea12f89153086746821cf6c9ef93ab88360a50c1aeefa1fe522421704",
    strip_prefix = "rules_nodejs-6.0.0-beta1",
    url = "https://github.com/bazelbuild/rules_nodejs/releases/download/v6.0.0-beta1/rules_nodejs-v6.0.0-beta1.tar.gz",
)

load("@rules_nodejs//nodejs:repositories.bzl", "DEFAULT_NODE_VERSION", "nodejs_register_toolchains")

nodejs_register_toolchains(
    name = "nodejs",
    node_version = DEFAULT_NODE_VERSION,
)

http_archive(
    name = "rules_tree_sitter",
    sha256 = "a09f177a2b8acb2f8a84def6ca0c41a5bd26b25634aa7313f22ade6c54e57ca1",
    strip_prefix = "rules_tree_sitter-bc3a2131053207de7dfd9b24046b811ce770e35d",
    urls = ["https://github.com/Maan2003/rules_tree_sitter/archive/bc3a2131053207de7dfd9b24046b811ce770e35d.tar.gz"],
)

load("@rules_tree_sitter//tree_sitter:tree_sitter.bzl", "tree_sitter_register_toolchains")

tree_sitter_register_toolchains()
