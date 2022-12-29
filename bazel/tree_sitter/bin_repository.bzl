# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sets up a repository with the tree-sitter binary (and nothing else).

Gather shas with:
    export RELEASE=####
    for v in linux-x64 macos-x64 windows-x64
    do
        echo "\"$v\": \"$(wget -q -O - https://github.com/tree-sitter/tree-sitter/releases/download/v$RELEASE/tree-sitter-$v.gz | sha256sum | cut -d ' ' -f1)\","
    done
"""

_URL = "https://github.com/tree-sitter/tree-sitter/releases/download/v{0}/tree-sitter-{1}.gz"

_RELEASE = "0.20.7"

_VERSION_SHAS = {
    "linux-x64": "633f47f2239ec45d320258da881a0d2bb2e4383e6b48a0d00713ed402f22d9b1",
    "macos-x64": "df5ec56221a78f009edba9e2b5842ef82cb2fd39b1357f8c7bc4b3963c61c652",
    "windows-x64": "77723edd739f1b4e7aa698e30f74c608a7a705f6fde876d58f98c587357e950e",
}

def _impl(repository_ctx):
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
        sha256 = _VERSION_SHAS[bin],
        url = _URL.format(_RELEASE, bin),
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

tree_sitter_bin_repository = repository_rule(implementation = _impl)
