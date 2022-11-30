# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for building fuzz tests."""

load("//bazel/cc_toolchains:defs.bzl", "cc_env")

def sh_run(name, args, **kwargs):
    """Produces a target which can run with the given args."""

    native.sh_binary(
        name = name,
        srcs = ["//bazel/sh_run:exec.sh"],
        args = args,
        **kwargs
    )

def glob_sh_run(file_exts, args, data, run_ext = "run", **kwargs):
    """Produces a per-file sh_run."""
    files = native.glob(
        ["**"],
        exclude_directories = 1,
    )
    for f in files:
        if f.split(".")[-1] not in file_exts:
            continue
        sh_run(
            name = "%s.%s" % (f, run_ext),
            args = args + ["$(location %s)" % f],
            data = data + [f],
            env = cc_env(),
            **kwargs
        )
