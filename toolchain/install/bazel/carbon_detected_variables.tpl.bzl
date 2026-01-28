# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A Starlark file exporting detected Carbon toolchain configuration variables.

This file gets processed by a repository rule, substituting the `VARIABLE`s with
values, for example using an invocation of `carbon config`.
"""

clang_include_dirs = CLANG_INCLUDE_DIRS
clang_sysroot = "CLANG_SYSROOT"
