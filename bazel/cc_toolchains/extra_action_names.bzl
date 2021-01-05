# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A Starlark file exporting feature configuration action names that are not defined in Bazel.
"""

EXTRA_ACTION_NAMES = struct(
    clang_format = "clang_format_action"
)
