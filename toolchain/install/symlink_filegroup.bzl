# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule for symlinking an entire filegroup, preserving its structure."""

def _symlink_filegroup_impl(ctx):
    prefix = ctx.attr.out_prefix
    remove_prefix = ctx.attr.remove_prefix

    outputs = []
    for f in ctx.files.srcs:
        out = ctx.actions.declare_file(
            prefix + f.short_path.removeprefix(remove_prefix),
        )
        outputs.append(out)
        ctx.actions.symlink(output = out, target_file = f)

    if len(ctx.files.srcs) != len(outputs):
        fail("Output count mismatch!")

    return [
        DefaultInfo(
            files = depset(outputs),
            runfiles = ctx.runfiles(files = outputs),
        ),
    ]

symlink_filegroup = rule(
    implementation = _symlink_filegroup_impl,
    attrs = {
        "out_prefix": attr.string(mandatory = True),
        "remove_prefix": attr.string(default = ""),
        "srcs": attr.label_list(mandatory = True),
    },
)
