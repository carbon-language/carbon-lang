# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Feature configuration action names that are not defined in Bazel."""

EXTRA_ACTION_NAMES = struct(
    clang_tidy = "clang_tidy_action",
)
