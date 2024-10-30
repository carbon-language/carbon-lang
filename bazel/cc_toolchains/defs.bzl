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

    # On macOS, there's a nano zone allocation warning due to asan (arises
    # in fastbuild/dbg). This suppresses the warning in `bazel run`.
    #
    # Concatenation of a dict with a select isn't supported, so we concatenate
    # within the select.
    # https://github.com/bazelbuild/bazel/issues/12457
    return select({
        "//bazel/cc_toolchains:macos_asan": {
            "LLVM_SYMBOLIZER_PATH": llvm_symbolizer,
            "MallocNanoZone": "0",
        },
        "//conditions:default": {
            "LLVM_SYMBOLIZER_PATH": llvm_symbolizer,
        },
    })
