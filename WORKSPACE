# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

workspace(name = "carbon")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

###############################################################################
# C++ rules
###############################################################################

# Configure the bootstrapped Clang and LLVM toolchain for Bazel.
load(
    "//bazel/cc_toolchains:clang_configuration.bzl",
    "clang_register_toolchains",
)

clang_register_toolchains(name = "bazel_cc_toolchain")

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
