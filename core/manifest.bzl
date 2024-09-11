# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule for producing a manifest for a filegroup."""

def _manifest(ctx):
    dir = ctx.build_file_path.removesuffix("BUILD")
    content = [f.path.removeprefix(dir) for f in ctx.files.srcs]
    ctx.actions.write(ctx.outputs.out, "\n".join(content) + "\n")

    return [
        DefaultInfo(
            files = depset(direct = [ctx.outputs.out]),
            default_runfiles = ctx.runfiles(files = [ctx.outputs.out]),
        ),
    ]

manifest = rule(
    implementation = _manifest,
    attrs = {
        "out": attr.output(mandatory = True),
        "srcs": attr.label_list(mandatory = True),
    },
)
