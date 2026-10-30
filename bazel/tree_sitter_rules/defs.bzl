# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Allows running tree-sitter generate."""

load("@bazel_skylib//rules:native_binary.bzl", "native_binary")

# values taken from Maven's OS detector
def get_os(repository_ctx):
    os = repository_ctx.os.name
    if os.startswith("linux"):
        return "linux"
    if os.startswith("mac") or os.startswith("osx"):
        return "macos"
    if os.startswith("freebsd"):
        return "freebsd"
    if os.startswith("openbsd"):
        return "openbsd"
    if os.startswith("netbsd"):
        return "netbsd"
    if os.startswith("windows"):
        return "windows"

    fail("Unknown OS: %s" % os)

def get_arch(repository_ctx):
    arch = repository_ctx.os.arch
    if arch in ("x86_32", "x86", "i386", "i486", "i586", "i686", "ia32", "x32"):
        return "x86"
    if arch in ("x86_64", "amd64", "ia32e", "em64t", "x64"):
        return "x64"
    if arch in ("arm", "arm32"):
        return "arm"
    if arch == "aarch64":
        return "arm64"

    fail("Unknown architecture: %s" % arch)

def _tree_sitter_binary_impl(repository_ctx):
    os = get_os(repository_ctx)
    arch = get_arch(repository_ctx)
    executable_suffix = ""

    # it's possible that it would be better to build the binary for ourselves
    if os == "linux" and arch == "x64":
        url = "https://github.com/tree-sitter/tree-sitter/releases/download/v0.27.0/tree-sitter-cli-linux-x64.zip"
        sha = "e4a3826bcd0fe099ee3a5617767374939cbc23c4a35b5b53f5fc04142525a2c1"
    elif os == "linux" and arch == "arm64":
        url = "https://github.com/tree-sitter/tree-sitter/releases/download/v0.27.0/tree-sitter-cli-linux-arm64.zip"
        sha = "6260b621bf5ab87027dfb463bf955504ef32cdcda62b81f28447753e48c83a62"
    elif os == "macos" and arch == "x64":
        url = "https://github.com/tree-sitter/tree-sitter/releases/download/v0.27.0/tree-sitter-cli-macos-x64.zip"
        sha = "4509c86918341ca50877ce20ef1507390257d203b4893f6c66b6f5cc632a61cd"
    elif os == "macos" and arch == "arm64":
        url = "https://github.com/tree-sitter/tree-sitter/releases/download/v0.27.0/tree-sitter-cli-macos-arm64.zip"
        sha = "f278063d8544160f6f89f7f8dba6ba112cb0dd1669757788d2bb7a8a613d2c58"
    elif os == "windows" and arch == "x64":
        url = "https://github.com/tree-sitter/tree-sitter/releases/download/v0.27.0/tree-sitter-cli-windows-x64.zip"
        sha = "de043e4c289efcef7981df99988300f23763befdeb36abf7c1898f080c478424"
        executable_suffix = ".exe"
    elif os == "windows" and arch == "arm64":
        url = "https://github.com/tree-sitter/tree-sitter/releases/download/v0.27.0/tree-sitter-cli-windows-arm64.zip"
        sha = "eadbb00c87c5a38f874eb92af593df7264635afefb455af3218bdb1c50123bad"
        executable_suffix = ".exe"
    else:
        fail("Unsupported OS/Architecture combination: %s/%s" % (os, arch))

    repository_ctx.download_and_extract(
        url = url,
        sha256 = sha,
    )
    repository_ctx.file("BUILD", """
load("@bazel_skylib//rules:native_binary.bzl", "native_binary")

native_binary(
    name = "tree_sitter",
    src = "tree-sitter%s",
    visibility = ["//visibility:public"],
)
""" % executable_suffix)

_tree_sitter_binary = repository_rule(
    implementation = _tree_sitter_binary_impl,
)
tree_sitter_binary = module_extension(
    implementation = lambda ctx: _tree_sitter_binary(name = "tree_sitter_binary"),
    os_dependent = True,
    arch_dependent = True,
)

def tree_sitter_parser(name, src):
    """Generates the tree sitter parser from the grammar.js"""
    native.genrule(
        name = name,
        tools = ["@tree_sitter_binary//:tree_sitter"],
        srcs = [src],
        outs = [
            "parser.c",
            "tree_sitter/parser.h",
        ],
        cmd = "$(execpath @tree_sitter_binary//:tree_sitter) generate $(execpath %s) --js-runtime native -o $(RULEDIR)" % src,
        tags = ["manual"],
    )
