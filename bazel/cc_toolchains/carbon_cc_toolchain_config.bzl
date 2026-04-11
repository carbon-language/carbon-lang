# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Starlark cc_toolchain configuration rules for using the Carbon toolchain"""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load(
    "@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "action_config",
    "flag_group",
    "flag_set",
    "tool",
)
load(
    "@rules_cc//cc:defs.bzl",
    "CcToolchainConfigInfo",
    "cc_toolchain",
)
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("//toolchain/runtimes:carbon_runtimes.bzl", "carbon_runtimes_build")
load(
    "carbon_clang_variables.bzl",
    "clang_include_dirs",
    "clang_resource_dir",
    "clang_sysroot",
)
load(
    "cc_toolchain_actions.bzl",
    "all_c_compile_actions",
    "all_cpp_compile_actions",
    "all_link_actions",
)
load("cc_toolchain_carbon_project_features.bzl", "carbon_project_features")
load("cc_toolchain_features.bzl", "clang_cc_toolchain_features")

def _make_action_configs(tools, runtimes_path = None):
    runtimes_flag = "--no-build-runtimes"
    if runtimes_path:
        runtimes_flag = "--prebuilt-runtimes={0}".format(runtimes_path)

    return [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tools.clang],
        )
        for name in all_c_compile_actions
    ] + [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tools.clangpp],
        )
        for name in all_cpp_compile_actions
    ] + [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tools.carbon_busybox],
            flag_sets = [flag_set(flag_groups = [flag_group(flags = [
                runtimes_flag,
                "link",
                # We want to allow Bazel to intermingle linked object files and
                # Clang-spelled link flags. The first `--` starts the list of
                # initial object files by ending flags to the `link` subcommand,
                # and the second `--` switches to Clang-spelled flags.
                "--",
                "--",
            ])])],
        )
        for name in all_link_actions
    ] + [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tools.llvm_ar],
        )
        for name in [ACTION_NAMES.cpp_link_static_library]
    ] + [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tools.llvm_strip],
        )
        for name in [ACTION_NAMES.strip]
    ]

def _compute_clang_system_include_dirs():
    system_include_dirs_start_index = None
    for index, dir in enumerate(clang_include_dirs):
        # Skip over the include search directories until we find the resource
        # directory. The system include directories are everything after that.
        if dir.startswith(clang_resource_dir):
            system_include_dirs_start_index = index + 1
            break
    if not system_include_dirs_start_index:
        fail("Could not find the resource directory in the clang include " +
             "directories: {}".format(clang_include_dirs))
    return clang_include_dirs[system_include_dirs_start_index:]

def _carbon_cc_toolchain_config_impl(ctx):
    tools = struct(
        carbon_busybox = tool(path = "carbon-busybox"),
        clang = tool(path = "llvm/bin/clang"),
        clangpp = tool(path = "llvm/bin/clang++"),
        llvm_ar = tool(path = "llvm/bin/llvm-ar"),
        llvm_strip = tool(path = "llvm/bin/llvm-strip"),
    )
    if ctx.attr.bins:
        carbon_busybox = None
        clang = None
        clangpp = None
        llvm_ar = None
        llvm_strip = None
        for f in ctx.files.bins:
            if f.basename == "carbon-busybox":
                carbon_busybox = f
            elif f.basename == "clang":
                clang = f
            elif f.basename == "clang++":
                clangpp = f
            elif f.basename == "llvm-ar":
                llvm_ar = f
            elif f.basename == "llvm-strip":
                llvm_strip = f
        tools = struct(
            carbon_busybox = tool(tool = carbon_busybox),
            clang = tool(tool = clang),
            clangpp = tool(tool = clangpp),
            llvm_ar = tool(tool = llvm_ar),
            llvm_strip = tool(tool = llvm_strip),
        )

    # Only use a sysroot if a non-trivial one is set in Carbon's config.
    builtin_sysroot = None
    sysroot_include_search = []
    if clang_sysroot != "None" and clang_sysroot != "/":
        builtin_sysroot = clang_sysroot
        sysroot_include_search = ["%sysroot%/usr/include"]

    runtimes_path = None
    if ctx.attr.runtimes:
        for f in ctx.files.runtimes:
            if f.basename == "runtimes_root":
                runtimes_path = f.dirname
                break
        if not runtimes_path:
            fail("Unable to compute the runtimes path for: {0}".format(
                ctx.attr.runtimes,
            ))

    identifier = "{0}_toolchain_{1}_{2}".format(
        ctx.attr.identifier_prefix,
        ctx.attr.target_cpu,
        ctx.attr.target_os,
    )
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = clang_cc_toolchain_features(
            target_os = ctx.attr.target_os,
            target_cpu = ctx.attr.target_cpu,

            # TODO: This should be configured externally rather than here so
            # that the install Carbon toolchain doesn't automatically include
            # Carbon-project-specific flags. However, that is especially awkward
            # to do until we fully migrate to a rules-based toolchain, and the
            # project-specific flags are largely harmless at the moment. We also
            # omit a meaningful cache key as when using the Carbon toolchain we
            # don't need it as it is a hermetic part of Bazel.
            project_features = carbon_project_features(cache_key = ""),
        ),
        action_configs = _make_action_configs(tools, runtimes_path),
        cxx_builtin_include_directories = [
            "runtimes/libunwind/include",
            "runtimes/libcxx/include",
            "runtimes/libcxxabi/include",
            "{}/include".format(clang_resource_dir),
            "runtimes/clang_resource_dir/include",
        ] + _compute_clang_system_include_dirs() + sysroot_include_search,
        builtin_sysroot = builtin_sysroot,

        # This configuration only supports local non-cross builds so derive
        # everything from the target CPU selected.
        toolchain_identifier = identifier,

        # This is used to expose a "flag" that `config_setting` rules can use to
        # determine if the compiler is Clang.
        compiler = "clang",
    )

carbon_cc_toolchain_config = rule(
    implementation = _carbon_cc_toolchain_config_impl,
    attrs = {
        "bins": attr.label(mandatory = False),
        "identifier_prefix": attr.string(mandatory = True),
        "runtimes": attr.label(mandatory = False),
        "target_cpu": attr.string(mandatory = True),
        "target_os": attr.string(mandatory = True),
    },
    provides = [CcToolchainConfigInfo],
)

def _set_runtimes_build_transition_impl(_, attr):
    return {
        "//:bootstrap_stage": attr.stage,
        "//:runtimes_build": True,
    }

set_runtimes_build_transition = transition(
    inputs = [],
    outputs = [
        "//:runtimes_build",
        "//:bootstrap_stage",
    ],
    implementation = _set_runtimes_build_transition_impl,
)

def _set_runtimes_build_filegroup_impl(ctx):
    return [DefaultInfo(files = depset(ctx.files.srcs))]

set_runtimes_build_filegroup = rule(
    implementation = _set_runtimes_build_filegroup_impl,
    attrs = {
        # Mark that our dependencies are built through a transition.
        "srcs": attr.label_list(mandatory = True, cfg = set_runtimes_build_transition),

        # The platform to use when building the runtimes.
        "stage": attr.int(mandatory = True),

        # Enable transitions in this rule.
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)

def _set_stage_transition_impl(_, attr):
    return {
        "//:bootstrap_stage": attr.stage,
        "//:runtimes_build": False,
    }

set_stage_transition = transition(
    inputs = [],
    outputs = [
        "//:bootstrap_stage",
        "//:runtimes_build",
    ],
    implementation = _set_stage_transition_impl,
)

def _set_stage_filegroup_impl(ctx):
    return [DefaultInfo(files = depset(ctx.files.srcs))]

set_stage_filegroup = rule(
    implementation = _set_stage_filegroup_impl,
    attrs = {
        # Mark that our dependencies are built through a transition.
        "srcs": attr.label_list(mandatory = True, cfg = set_stage_transition),

        # The stage to use when building the toolchain.
        "stage": attr.int(mandatory = True),

        # Enable transitions in this rule.
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)

def carbon_cc_toolchain_suite(
        name,
        all_hdrs,
        base_files,
        clang_hdrs,
        platforms,
        runtimes_cfg,
        build_stage = 1,
        base_stage = 0,
        tags = []):
    """Create a Carbon `cc_toolchain` for the current target platform.

    This provides the final toolchain for Carbon, but also all of the
    infrastructure for supporting on-demand built runtimes in this toolchain.

    There is also support for bootstrapping, where one `build_stage` toolchain
    builds on top of another `base_stage`.

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

    # First, declare file groups that explicitly built using the base stage, and
    # not in the runtimes build. These allow us to form the inputs to both the
    # runtimes toolchain and the main toolchain of this stage that are built
    # entirely by the base toolchain.
    set_stage_filegroup(
        name = "{}_clang_hdrs".format(name),
        srcs = clang_hdrs,
        stage = base_stage,
        tags = tags,
    )

    set_stage_filegroup(
        name = "{}_base_files".format(name),
        srcs = base_files,
        stage = base_stage,
        tags = tags,
    )

    set_stage_filegroup(
        name = "{}_runtimes_compile_files".format(name),
        srcs = [
            ":{}_base_files".format(name),
            ":{}_clang_hdrs".format(name),
        ],
        stage = base_stage,
        tags = tags,
    )

    set_stage_filegroup(
        name = "{}_compile_files".format(name),
        srcs = [":{}_base_files".format(name)] + all_hdrs,
        stage = base_stage,
        tags = tags,
    )

    # Now build a configuration and toolchain that is configured to work
    # _without_ runtimes, and be used to _build_ the runtimes on-demand.
    carbon_cc_toolchain_config(
        name = "{}_runtimes_toolchain_config".format(name),
        identifier_prefix = "{}_runtimes".format(name),
        target_cpu = select({
            # Note that we need to select on both OS and CPU so that we end up
            # spelling the CPU in the correct OS-specific ways.
            ":is_{}_{}".format(os, cpu): cpu
            for os, cpus in platforms.items()
            for cpu in cpus
        }),
        target_os = select({
            "@platforms//os:{}".format(os): os
            for os in platforms.keys()
        }),
        bins = ":{}_base_files".format(name),
        tags = tags,
    )

    cc_toolchain(
        name = "{}_runtimes_cc_toolchain".format(name),
        all_files = ":{}_runtimes_compile_files".format(name),
        ar_files = ":{}_base_files".format(name),
        as_files = ":{}_runtimes_compile_files".format(name),
        compiler_files = ":{}_runtimes_compile_files".format(name),
        dwp_files = ":{}_base_files".format(name),
        linker_files = ":{}_base_files".format(name),
        objcopy_files = ":{}_base_files".format(name),
        strip_files = ":{}_base_files".format(name),
        toolchain_config = ":{}_runtimes_toolchain_config".format(name),
        toolchain_identifier = select({
            ":is_{}_{}".format(os, cpu): "{}_{}_{}_runtimes_toolchain".format(name, os, cpu)
            for os, cpus in platforms.items()
            for cpu in cpus
        }),
        tags = tags,
    )

    native.toolchain(
        name = "{}_runtimes_toolchain".format(name),
        target_settings = [
            ":is_bootstrap_stage_{}".format(build_stage),
            ":is_runtimes_build",
        ],
        use_target_platform_constraints = True,
        toolchain = ":{}_runtimes_cc_toolchain".format(name),
        toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
        tags = tags,
    )

    # Now that we have a toolchain for building runtimes, actually do the build
    # here using the runtimes config provided to us. This is important to do
    # here because we need each runtimes build for a particular stage of
    # toolchain to be distinct.
    carbon_runtimes_build(
        name = "{}_runtimes_build".format(name),
        config = runtimes_cfg,
        clang_hdrs = [":{}_clang_hdrs".format(name)],
        tags = tags,
    )

    # Wrap the built runtimes for this stage in a filegroup that enforces that
    # the runtimes toolchain for this stage.
    set_runtimes_build_filegroup(
        name = "{}_runtimes".format(name),
        srcs = ["{}_runtimes_build".format(name)],
        stage = build_stage,
        tags = tags,
    )

    # Now we can build the main toolchain configuration, filegroups including
    # the on-demand built runtimes, and the final tolochain itself.
    carbon_cc_toolchain_config(
        name = "{}_toolchain_config".format(name),
        identifier_prefix = name,
        target_cpu = select({
            # Note that we need to select on both OS and CPU so that we end up
            # spelling the CPU in the correct OS-specific ways.
            ":is_{}_{}".format(os, cpu): cpu
            for os, cpus in platforms.items()
            for cpu in cpus
        }),
        target_os = select({
            "@platforms//os:{}".format(os): os
            for os in platforms.keys()
        }),
        runtimes = ":{}_runtimes".format(name),
        bins = ":{}_base_files".format(name),
        tags = tags,
    )

    native.filegroup(
        name = "{}_linker_files".format(name),
        srcs = [
            ":{}_base_files".format(name),
            ":{}_runtimes".format(name),
        ],
        tags = tags,
    )

    native.filegroup(
        name = "{}_all_files".format(name),
        srcs = [
            ":{}_compile_files".format(name),
            ":{}_linker_files".format(name),
        ],
        tags = tags,
    )

    cc_toolchain(
        name = "{}_cc_toolchain".format(name),
        all_files = ":{}_all_files".format(name),
        ar_files = ":" + name + "_base_files",
        as_files = ":" + name + "_compile_files",
        compiler_files = ":" + name + "_compile_files",
        dwp_files = ":" + name + "_linker_files",
        linker_files = ":" + name + "_linker_files",
        objcopy_files = ":" + name + "_base_files",
        strip_files = ":" + name + "_base_files",
        toolchain_config = ":" + name + "_toolchain_config",
        toolchain_identifier = select({
            ":is_{}_{}".format(os, cpu): "{}_{}_{}_toolchain".format(name, os, cpu)
            for os, cpus in platforms.items()
            for cpu in cpus
        }),
        tags = tags,
    )

    native.toolchain(
        name = name + "_toolchain",
        target_settings = [":is_bootstrap_stage_{}".format(build_stage), ":not_runtimes_build"],
        use_target_platform_constraints = True,
        toolchain = ":" + name + "_cc_toolchain",
        toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
        tags = tags,
    )
