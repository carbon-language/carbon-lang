# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provides helpers for cc rules. Intended for general consumption."""

# The hermetic llvm-symbolizer target.
_llvm_symbolizer = "@llvm-project//llvm:llvm-symbolizer"

def cc_env():
    """Returns standard environment settings for a cc_binary.

    In use, this should set both `data` and `env`, as in:

    ```
    load("//bazel/cc_toolchains:defs.bzl", "cc_env", "cc_env_data")

    cc_binary(
      ...
      data = cc_env_data(),
      env = cc_env(),
    )
    ```

    We're currently setting this on a target-by-target basis, mainly because
    it's difficult to modify default behaviors.
    """
    env = {"LLVM_SYMBOLIZER_PATH": "$(location {0})".format(_llvm_symbolizer)}

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

def cc_env_data():
    """Returns data needed for cc_env().

    Set up as a function mainly for parity, and in case we need future changes.
    """
    return [_llvm_symbolizer]
