# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for symlinking in ways that assist install_filegroups."""

_SYMLINK_BUSYBOX_TMPL = """#!/usr/bin/env python3

from pathlib import Path
import os
import sys

_RELATIVE_PATH = "{0}"
_BUSYBOX_ARGS = {1}

# Run the tool using the absolute path, forwarding arguments.
tool_path = Path(__file__).parent / _RELATIVE_PATH
os.execv(tool_path, [tool_path] + _BUSYBOX_ARGS + sys.argv[1:])
"""

def _busybox_wrapper_impl(ctx):
    """Symlinking busybox things needs special logic.

    This is because Bazel doesn't cache the actual symlink, resulting in
    essentially resolved symlinks being produced in place of the expected tool.
    As a consequence, we can't rely on the symlink name when dealing with
    busybox entries.

    An example repro of this using a local build cache is:
        bazel build //toolchain
        bazel clean
        bazel build //toolchain

    We could in theory get reasonable behavior with
    `ctx.actions.declare_symlink`, but that's disallowed in our `.bazelrc` for
    cross-environment compatibility.

    The particular approach here uses the Python script as a launching pad so
    that the busybox still receives an appropriate location in argv[0], allowing
    it to find other files in the lib directory. Arguments are inserted to get
    equivalent behavior as if symlink resolution had occurred.

    The underlying bug is noted at:
    https://github.com/bazelbuild/bazel/issues/23620
    """
    content = _SYMLINK_BUSYBOX_TMPL.format(
        ctx.attr.symlink,
        ctx.attr.busybox_args,
    )
    ctx.actions.write(
        output = ctx.outputs.executable,
        is_executable = True,
        content = content,
    )
    return []

busybox_wrapper = rule(
    doc = "Helper for running a busybox with symlink-like characteristics.",
    implementation = _busybox_wrapper_impl,
    attrs = {
        "busybox_args": attr.string_list(
            doc = "Optional arguments to pass for equivalent behavior to a symlink.",
        ),
        "symlink": attr.string(mandatory = True),
    },
    executable = True,
)

def _symlink_file_impl(ctx):
    executable = None
    if ctx.attr.symlink_binary:
        out = ctx.actions.declare_file(ctx.label.name)
        ctx.actions.symlink(
            output = out,
            target_file = ctx.file.symlink_binary,
            is_executable = True,
        )
        executable = out
    elif ctx.attr.symlink_label:
        out = ctx.actions.declare_file(ctx.label.name)
        ctx.actions.symlink(
            output = out,
            target_file = ctx.file.symlink_label,
        )
    else:
        fail("Missing symlink target")

    return [
        DefaultInfo(
            executable = executable,
            files = depset(direct = [out]),
            default_runfiles = ctx.runfiles(files = [out]),
        ),
    ]

symlink_file = rule(
    doc = "Symlinks a single file, with support for multiple approaches.",
    implementation = _symlink_file_impl,
    attrs = {
        "symlink_binary": attr.label(
            allow_single_file = True,
            executable = True,
            cfg = "target",
        ),
        "symlink_label": attr.label(allow_single_file = True),
    },
)

def _symlink_filegroup_impl(ctx):
    prefix = ctx.attr.out_prefix

    outputs = []
    for f in ctx.files.srcs:
        # We normalize the path to be package-relative in order to ensure
        # consistent paths across possible repositories.
        relative_path = f.short_path.removeprefix(f.owner.package)

        out = ctx.actions.declare_file(prefix + relative_path)
        outputs.append(out)
        ctx.actions.symlink(output = out, target_file = f)

    if len(ctx.files.srcs) != len(outputs):
        fail("Output count mismatch!")

    return [
        DefaultInfo(
            files = depset(direct = outputs),
            default_runfiles = ctx.runfiles(files = outputs),
        ),
    ]

symlink_filegroup = rule(
    doc = "Symlinks an entire filegroup, preserving its structure",
    implementation = _symlink_filegroup_impl,
    attrs = {
        "out_prefix": attr.string(mandatory = True),
        "srcs": attr.label_list(mandatory = True),
    },
)
