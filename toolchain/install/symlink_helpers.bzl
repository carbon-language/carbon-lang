# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for symlinking in ways that assist install_filegroups."""

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
    remove_prefix = ctx.attr.remove_prefix

    outputs = []
    for f in ctx.files.srcs:
        # We normalize the path to be package-relative in order to ensure
        # consistent paths across possible repositories. After experimenting,
        # incrementally removing these path fragments, in this order,
        # effectively handles the full permutation of files across different
        # repositories and both generated and source files.
        relative_path = f.path
        relative_path = relative_path.removeprefix(f.root.path + "/")
        relative_path = relative_path.removeprefix(f.owner.workspace_root + "/")
        relative_path = relative_path.removeprefix(f.owner.package + "/")
        if not relative_path.startswith(remove_prefix):
            fail("Missing `{0}` in `{1}`".format(remove_prefix, relative_path))
        relative_path = relative_path.removeprefix(remove_prefix)

        out = ctx.actions.declare_file(prefix + "/" + relative_path)
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
        "remove_prefix": attr.string(),
        "srcs": attr.label_list(mandatory = True),
    },
)
