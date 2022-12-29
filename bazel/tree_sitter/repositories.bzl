# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sets up a repository with the tree-sitter binary (and nothing else).

Gather shas with:
    export VERSION=####
    wget -q -O - https://github.com/tree-sitter/tree-sitter/archive/refs/tags/v$VERSION.tar.gz | sha256sum | cut -d ' ' -f1
    for p in linux-x64 macos-x64 windows-x64
    do
        echo "\"$p\": \"$(wget -q -O - https://github.com/tree-sitter/tree-sitter/releases/download/v$VERSION/tree-sitter-$p.gz | sha256sum | cut -d ' ' -f1)\","
    done
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# The version in use.
_version = "0.20.7"

# URLs for the source and binary downloads.
_source_url = "https://github.com/tree-sitter/tree-sitter/archive/refs/tags/v{0}.tar.gz"
_bin_url = "https://github.com/tree-sitter/tree-sitter/releases/download/v{0}/tree-sitter-{1}.gz"

# Checksums of source and binaries.
_source_sha = "b355e968ec2d0241bbd96748e00a9038f83968f85d822ecb9940cbe4c42e182e"
_bin_shas = {
    "linux-x64": "633f47f2239ec45d320258da881a0d2bb2e4383e6b48a0d00713ed402f22d9b1",
    "macos-x64": "df5ec56221a78f009edba9e2b5842ef82cb2fd39b1357f8c7bc4b3963c61c652",
    "windows-x64": "77723edd739f1b4e7aa698e30f74c608a7a705f6fde876d58f98c587357e950e",
}

def _bin_repo_impl(repository_ctx):
    """Sets up the binary repository.

    This is separated from source because it's a custom download, and the
    separation should make errors easier to diagnose.
    """
    os = repository_ctx.os.name
    if os.startswith("linux"):
        bin = "linux-x64"
    elif os.startswith("mac os"):
        bin = "macos-x64"
    elif os.startswith("windows"):
        bin = "windows-x64"
    else:
        fail("Unrecognized OS: {0}".format(os))

    repository_ctx.download(
        output = "tree-sitter.gz",
        sha256 = _bin_shas[bin],
        url = _bin_url.format(_version, bin),
    )

    # Bazel doesn't support extracting .gz, so we subprocess for it.
    result = repository_ctx.execute(
        ["gunzip", "tree-sitter.gz"],
    )
    if result.return_code != 0:
        fail(result.stderr)

    result = repository_ctx.execute(
        ["chmod", "+x", "tree-sitter"],
    )
    if result.return_code != 0:
        fail(result.stderr)

    repository_ctx.file("BUILD", "exports_files(['tree-sitter'])")

def tree_sitter_repositories():
    """Sets up the source and binary repositories."""
    bin_repo = repository_rule(implementation = _bin_repo_impl)
    bin_repo(name = "tree_sitter_bin")

    http_archive(
        name = "tree_sitter",
        sha256 = _source_sha,
        urls = [_source_url.format(_version)],
        strip_prefix = "tree-sitter-{0}".format(_version),
        build_file = "//bazel/tree_sitter:BUILD.tree_sitter",
    )
