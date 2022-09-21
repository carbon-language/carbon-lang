# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A Starlark file exporting detected Clang configuration variables.

This file gets processed by a repository rule, substituting the
`{VARIABLE}`s with values detected by invoking Clang.
"""

llvm_bindir = "{LLVM_BINDIR}"
clang_bindir = "{CLANG_BINDIR}"
clang_version = {CLANG_VERSION}
clang_version_for_cache = "{CLANG_VERSION_FOR_CACHE}"
clang_resource_dir = "{CLANG_RESOURCE_DIR}"
clang_include_dirs_list = {CLANG_INCLUDE_DIRS_LIST}
sysroot_dir = "{SYSROOT}"
