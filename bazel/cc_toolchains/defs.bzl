# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provides helpers for cc rules. Intended for general consumption."""

load("@bazel_cc_toolchain//:clang_detected_variables.bzl", "llvm_symbolizer")

def cc_env():
    """Returns standard environment settings for a cc_binary."""
    env = {"LLVM_SYMBOLIZER_PATH": llvm_symbolizer}

    # On macOS, there's a nano zone allocation warning due to asan (arises
    # in fastbuild/dbg). This suppresses the warning in `bazel run`.
    #
    # Concatenation of a dict with a select isn't supported, so we concatenate
    # within the select.
    # https://github.com/bazelbuild/bazel/issues/12457
    return select({
        "//bazel/cc_toolchains:macos_asan": env.update({"MallocNanoZone": "0"}),
        "//conditions:default": env,
    })
