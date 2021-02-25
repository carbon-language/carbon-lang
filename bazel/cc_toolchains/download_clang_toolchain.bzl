# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A Starlark rule to download prebuilt Clang toolchains"""

def _impl(repository_ctx):
    """Downloads and extracts an LLVM build for the current platform."""
    os = repository_ctx.os.name
    if os == 'linux':
        url = 'https://github.com/mmdriley/llvm-builds/releases/download/r26/llvm-linux.tar.xz'
        sha256 = '4b6c94f3e0c1431b2211c3e76ad225b118715d26f813e4e180a47799730190ab'
    elif os == 'mac os x':
        url = 'https://github.com/mmdriley/llvm-builds/releases/download/r26/llvm-macos.tar.xz'
        sha256 = '53c99046baef90eef7b2491f653c1e338668ebf7abe98880798c91acef9ddc2d'
    elif os.startswith('windows'):
        url = 'https://github.com/mmdriley/llvm-builds/releases/download/r26/llvm-windows.tar.xz'
        sha256 = '86e4dc869d6a84ca6f3de489b772d8d5ff8b2757d92f2135496aa88493274d5f'
    else:
        fail('no clang distribution to download for os: ' + os)

    repository_ctx.download_and_extract(url, sha256=sha256)

    # Make sure there's an (empty) BUILD file in the root so we can directly
    # refer to binaries, e.g. `@repo//:bin/clang`.
    repository_ctx.file('BUILD')


download_clang_toolchain = repository_rule(
    implementation = _impl,
)
