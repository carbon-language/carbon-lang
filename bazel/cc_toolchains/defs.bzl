# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provides helpers for cc rules. Intended for general consumption."""

load("@bazel_cc_toolchain//:clang_detected_variables.bzl", "llvm_symbolizer")

def cc_env():
    """Returns standard environment settings for a cc_binary.

    In use, this looks like:

    ```
    load("//bazel/cc_toolchains:defs.bzl", "cc_env")

    cc_binary(
      ...
      env = cc_env(),
    )
    ```

    We're currently setting this on a target-by-target basis, mainly because
    it's difficult to modify default behaviors.
    """

    # Settings which apply cross-platform.
    # buildifier: disable=unsorted-dict-items
    common_env = {
        "LLVM_SYMBOLIZER_PATH": llvm_symbolizer,
        # Sanitizers don't use LLVM as fallback, but sometimes ASAN may be used
        # for UBSAN errors; we still set UBSAN in case it's directly used.
        "ASAN_SYMBOLIZER_PATH": llvm_symbolizer,
        "UBSAN_SYMBOLIZER_PATH": llvm_symbolizer,
        # Default to printing traces for UBSAN.
        "UBSAN_OPTIONS": "print_stacktrace=1",
    }

    # On macOS, there's a nano zone allocation warning when asan is enabled.
    # This suppresses the warning in `bazel run`.
    macos_env = {"MallocNanoZone": "0"}

    return common_env | select({
        "//bazel/cc_toolchains:macos_asan": macos_env,
        "//conditions:default": {},
    })
