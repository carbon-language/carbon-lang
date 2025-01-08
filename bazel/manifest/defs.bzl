# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule for producing a manifest for a filegroup."""

def _manifest(ctx):
    out = ctx.actions.declare_file(ctx.label.name)

    files = []
    for src in ctx.attr.srcs:
        files.extend([f.path for f in src[DefaultInfo].files.to_list()])
        files.extend([
            f.path
            for f in src[DefaultInfo].default_runfiles.files.to_list()
        ])

    if ctx.attr.strip_package_dir:
        package_dir = ctx.label.package + "/"
        content = [f.removeprefix(package_dir) for f in files]
    else:
        content = files

    ctx.actions.write(out, "\n".join(content) + "\n")

    return [
        DefaultInfo(
            files = depset(direct = [out]),
            runfiles = ctx.runfiles(files = [out]),
        ),
    ]

manifest = rule(
    implementation = _manifest,
    attrs = {
        "srcs": attr.label_list(allow_files = True, mandatory = True),
        "strip_package_dir": attr.bool(default = False),
    },
)
