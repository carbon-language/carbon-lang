# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provides support for macOS-specific malloc issues."""

def macos_malloc_env():
    """Disable nano malloc when running asan.

    On macOS, there's a nano zone allocation warning due to asan (arises
    in fastbuild/dbg). This suppresses the warning in `bazel run`.
    """
    return select({
        "//bazel/macos_malloc:macos_asan": {"MallocNanoZone": "0"},
        "//conditions:default": {},
    })
