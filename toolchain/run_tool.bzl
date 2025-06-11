# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Supports running a tool from the install filegroup."""

def _run_tool_impl(ctx):
    tool_files = ctx.attr.tool.files.to_list()
    if len(tool_files) != 1:
        fail("Expected 1 tool file, found {0}".format(len(tool_files)))
    ctx.actions.symlink(
        output = ctx.outputs.executable,
        target_file = tool_files[0],
        is_executable = True,
    )
    return [
        DefaultInfo(
            runfiles = ctx.runfiles(files = ctx.files.data),
        ),
        RunEnvironmentInfo(
            environment = ctx.attr.env |
                          {"CARBON_ARGV0_OVERRIDE": tool_files[0].short_path},
        ),
    ]

run_tool = rule(
    doc = "Helper for running a tool in a filegroup.",
    implementation = _run_tool_impl,
    attrs = {
        "data": attr.label_list(allow_files = True),
        "env": attr.string_dict(),
        "tool": attr.label(
            allow_single_file = True,
            executable = True,
            cfg = "target",
            mandatory = True,
        ),
    },
    executable = True,
)
