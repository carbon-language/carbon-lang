# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule to expand Bazel templates with version and build information.

This rule takes a source code template and turns that into a specific source
code output, substituting version information and build information from Bazel's
`stable-status.txt` and `volatile-status.txt` produced by the
`workpsace_status_command` during the build. When stamping is disabled, the
build information is replaced with constant values to provide better caching.

The template files should use Python's "template strings" syntax[1]. These rules
provide a fixed set of keys whose values will be substituted, and those keys
will always be substituted with something. They will have the value in the
stable status file if present, otherwise the value in the volatile status file
if present, otherwise the value "unknown". When reading keys from the status
files, a prefix of `STABLE_` will be removed from the key if present.

[1]: https://docs.python.org/3/library/string.html#template-strings

The substituted keys, and any guidance on values:
- `VERSION` (the version string for Carbon)
- `BUILD_EMBED_LABEL` (value of --embed_label)
- `BUILD_HOST` (the name of the host machine running the build)
- `BUILD_USER` (the name of the user running the build)
- `GIT_COMMIT_SHA` (output of `git parse-rev --short HEAD` or `unknown`)
- `GIT_DIRTY_SUFFIX` (`.dirty` if dirty client state or `` if unknown)
- `BUILD_TIMESTAMP` (the time of the build in seconds since the Unix Epoch)
- `ATTRIBUTE` (an optional attribute to apply to definitions, defaults to empty)
"""

load(":compute_version.bzl", "compute_version")

_STAMP_DOC = """
Follows behavior of the common 'stamp' attributes on rules. Set to 1 or 0 to
force stamping with actual build info on or off respectively, and to -1 to
follow the value of the command line flag `--stamp`.
"""

def _is_exec_config(ctx):
    """Detect if this is the exec configuration, previously known as "host".

    Sadly, there is not yet a supported way to detect this so replicate the
    hacks others currently use. Bazel issue:
    https://github.com/bazelbuild/bazel/issues/14444
    """
    return "-exec" in ctx.bin_dir.path or "/host/" in ctx.bin_dir.path

def _expand_version_build_info_impl(ctx):
    """Generates a file from a template, substituting version and build info."""
    inputs = [ctx.file.template]

    # The substitutions provided and their default values.
    substitutions = {
        "BUILD_EMBED_LABEL": "unknown",
        "BUILD_HOST": "unknown",
        "BUILD_TIMESTAMP": "unknown",
        "BUILD_USER": "unknown",
        "GIT_COMMIT_SHA": "unknown",
        "GIT_DIRTY_SUFFIX": "",
        "MAKE_WEAK": "0",
        "VERSION": compute_version(ctx),
    }
    substitutions.update(ctx.attr.substitutions)

    arguments = [
        "--template=" + ctx.file.template.path,
        "--output=" + ctx.outputs.out.path,
    ] + [
        "--substitution=" + key + "=" + value
        for key, value in substitutions.items()
    ]

    # We only want to allow stamping outside of the exec configuration.
    if not _is_exec_config(ctx):
        # Look at the attribute.
        stamp = ctx.attr.stamp

        # If requested, use the command line flag to select.
        if stamp == -1:
            # Set the default from `--stamp` / `--nostamp` command line flag,
            # which we detect through a macro and `config_setting`, and pipe
            # through an attribute.
            stamp = 1 if ctx.attr.internal_stamp_flag_detect else 0

        # Add the status files if stamping.
        if stamp == 1:
            inputs += [
                ctx.info_file,
                ctx.version_file,
            ]
            arguments += [
                "--status-file=" + ctx.info_file.path,
                "--status-file=" + ctx.version_file.path,
            ]

    ctx.actions.run(
        inputs = inputs,
        outputs = [ctx.outputs.out],
        executable = ctx.executable._gen_tmpl_tool,
        arguments = arguments,
        progress_message = "Generating templated source file: " +
                           ctx.outputs.out.short_path,
    )

expand_version_build_info_internal = rule(
    implementation = _expand_version_build_info_impl,
    attrs = {
        "internal_stamp_flag_detect": attr.bool(default = False),
        "out": attr.output(mandatory = True),
        "stamp": attr.int(values = [-1, 0, 1], default = -1, doc = _STAMP_DOC),
        "substitutions": attr.string_dict(
            doc = "Extra substitutions, potentially overriding defaults.",
        ),
        "template": attr.label(
            allow_single_file = True,
        ),
        "_gen_tmpl_tool": attr.label(
            default = Label("//bazel/version:gen_tmpl"),
            allow_files = True,
            executable = True,
            cfg = "exec",
        ),
        "_nightly_date_flag": attr.label(default = ":nightly_date"),
        "_pre_release_flag": attr.label(default = ":pre_release"),
        "_rc_number_flag": attr.label(default = ":rc_number"),
        "_release_flag": attr.label(default = ":release"),
    },
)

# We need a macro wrapping the rule so that we can inject a select that observes
# the `--stamp` command-line value and use that when needed.
def expand_version_build_info(name, **kwargs):
    expand_version_build_info_internal(
        name = name,
        internal_stamp_flag_detect = select({
            "//bazel/version:internal_stamp_flag_detect": True,
            "//conditions:default": False,
        }),
        **kwargs
    )
