# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Starlark rules for bootstrapping the Carbon toolchain."""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("//toolchain/runtimes:carbon_runtimes.bzl", "carbon_runtimes_build")
load(
    ":carbon_cc_toolchain_config.bzl",
    "carbon_cc_toolchain",
)

def _bootstrap_transition_impl(_, attr):
    return {
        "//:bootstrap_stage": attr.stage,

        # Note that we need to either set or clear the runtimes build flag each
        # time we transition to a different bootstarp stage or we can
        # incorrectly inherit an unexpected state.
        "//:runtimes_build": attr.enable_runtimes_build,
    }

_bootstrap_transition = transition(
    inputs = [],
    outputs = [
        "//:bootstrap_stage",
        "//:runtimes_build",
    ],
    implementation = _bootstrap_transition_impl,
)

def _filegroup_with_stage_impl(ctx):
    return [DefaultInfo(files = depset(ctx.files.srcs))]

filegroup_with_stage = rule(
    implementation = _filegroup_with_stage_impl,
    attrs = {
        "enable_runtimes_build": attr.bool(default = False),
        "srcs": attr.label_list(mandatory = True, cfg = _bootstrap_transition),
        "stage": attr.int(mandatory = True),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
    doc = "A filegroup whose sources are built using a specific toolchain stage.",
)

def _exec_filegroup_impl(ctx):
    return [DefaultInfo(files = depset(ctx.files.srcs))]

_exec_filegroup = rule(
    implementation = _exec_filegroup_impl,
    attrs = {
        "srcs": attr.label_list(cfg = "exec"),
    },
)

def filegroup_with_stage_and_exec(name, srcs, stage, tags = []):
    """Wraps `filegroup_with_stage` with a conditional `exec` config transition.

    When `//:bootstrap_exec_config` is disabled, this works exactly like
    `filegroup_with_stage`. But when it is _enabled_, it also adds an `exec`
    config transition.
    """
    impl_tags = tags if "manual" in tags else tags + ["manual"]

    filegroup_with_stage(
        name = name + "_stage_only",
        srcs = srcs,
        stage = stage,
        tags = impl_tags,
    )

    _exec_filegroup(
        name = name + "_with_exec",
        srcs = [":" + name + "_stage_only"],
        tags = impl_tags,
    )

    native.alias(
        name = name,
        actual = select({
            "//:bootstrap_with_exec_config": ":" + name + "_with_exec",
            "//conditions:default": ":" + name + "_stage_only",
        }),
        tags = tags,
    )

def _gen_cc_toolchain_paths_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)

    expanded_vars = [
        ctx.expand_make_variables("vars", v, {})
        for v in ctx.attr.vars
    ]

    out = ctx.actions.declare_file(ctx.attr.name + ".txt")
    ctx.actions.write(out, "\n".join(expanded_vars) + "\n")

    # Include all toolchain files in runfiles.
    runfiles = ctx.runfiles(files = [out]).merge(
        ctx.runfiles(transitive_files = cc_toolchain.all_files),
    )

    return [DefaultInfo(files = depset([out]), runfiles = runfiles)]

gen_cc_toolchain_paths_with_stage = rule(
    implementation = _gen_cc_toolchain_paths_impl,
    attrs = {
        "enable_runtimes_build": attr.bool(default = False),
        "stage": attr.int(mandatory = True),
        "vars": attr.string_list(
            default = ["$(CC)", "$(AR)", "$(NM)", "$(OBJCOPY)", "$(STRIP)"],
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    cfg = _bootstrap_transition,
)

def carbon_bootstrapped_cc_toolchain(
        name,
        all_hdrs,
        base_files,
        clang_hdrs,
        platforms,
        runtimes_cfg,
        build_stage = 1,
        base_stage = 0,
        tags = []):
    """Create a bootstrapped Carbon `cc_toolchain` for the current target.

    This builds on `carbon_cc_toolchain`, but enables bootstrapping the produced
    toolchain from a base stage's toolchain.

    Args:
        name:
            The name of the toolchain suite to produce, used as the base of the
            names of each component of the toolchain suite.
        all_hdrs: A list of header files to include in the toolchain.
        base_files: A list of files to include in the toolchain.
        build_stage: The stage to use for the build files.
        base_stage: The stage to use for the base files.
        clang_hdrs: A list of header files to include in the toolchain.
        platforms: An array of (os, cpu) pairs to support in the toolchain.
        runtimes_cfg: The runtimes configuration to use in the toolchain.
        tags: Tags to apply to the toolchain.
    """
    impl_tags = tags if "manual" in tags else tags + ["manual"]

    filegroup_with_stage_and_exec(
        name = "{}_clang_hdrs".format(name),
        srcs = clang_hdrs,
        stage = base_stage,
        tags = impl_tags,
    )

    filegroup_with_stage_and_exec(
        name = "{}_base_files".format(name),
        srcs = base_files,
        stage = base_stage,
        tags = impl_tags,
    )

    filegroup_with_stage_and_exec(
        name = "{}_runtimes_compile_files".format(name),
        srcs = [
            ":{}_base_files".format(name),
            ":{}_clang_hdrs".format(name),
        ],
        stage = base_stage,
        tags = impl_tags,
    )

    filegroup_with_stage_and_exec(
        name = "{}_compile_files".format(name),
        srcs = [":{}_base_files".format(name)] + all_hdrs,
        stage = base_stage,
        tags = impl_tags,
    )

    # The runtimes build for this stage of the bootstrap is only compatible with
    # both the build stage and the runtimes build. We'll induce those below, and
    # constrain them here to avoid any other usage.
    carbon_runtimes_build(
        name = "{}_runtimes_build".format(name),
        config = runtimes_cfg,
        clang_hdrs = ["{}_clang_hdrs".format(name)],
        tags = impl_tags,
    )

    # Wrap the runtimes build in a filegroup that both sets the stage to the
    # build stage as well as enabling runtimes building. Note that this is _not_
    # the base stage -- runtimes should be built by the same stage, simply using
    # the runtimes build setting.
    filegroup_with_stage(
        name = "{}_runtimes".format(name),
        srcs = [":{}_runtimes_build".format(name)],
        stage = build_stage,
        enable_runtimes_build = True,
        tags = impl_tags,
    )

    carbon_cc_toolchain(
        name = name,
        platforms = platforms,
        base_files_target = ":{}_base_files".format(name),
        runtimes_compile_files_target = ":{}_runtimes_compile_files".format(name),
        compile_files_target = ":{}_compile_files".format(name),
        runtimes_target = ":{}_runtimes".format(name),
        extra_toolchain_settings = [":is_bootstrap_stage_{}".format(build_stage)],
        tags = tags,
    )
