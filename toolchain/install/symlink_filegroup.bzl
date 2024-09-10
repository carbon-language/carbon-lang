# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule for symlinking an entire filegroup, preserving its structure."""

def _symlink_filegroup_impl(ctx):
    prefix = ctx.attr.out_prefix

    outputs = []
    for f in ctx.files.srcs:
        out = ctx.actions.declare_file(prefix + f.short_path)
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
    implementation = _symlink_filegroup_impl,
    attrs = {
        "out_prefix": attr.string(mandatory = True),
        "srcs": attr.label_list(mandatory = True),
    },
)
