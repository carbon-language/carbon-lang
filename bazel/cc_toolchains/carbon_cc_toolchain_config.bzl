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
            clang_resource_dir + "/include",
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

def _set_platform_transition_impl(settings, attr):
    original_platforms = settings["//:original_platforms"]

    # If the requested platform is the special value of the setting where we
    # store the original platforms on an initial transition, set the platform to
    # the saved list and clear it. Otherwise, we will set the platform to the
    # requested one.
    if not attr.platform:
        return {
            "//:original_platforms": [],
            "//command_line_option:platforms": original_platforms,
        }

    if original_platforms:
        # If there is already a saved original platforms list, preserve it.
        original_platforms = [str(label) for label in original_platforms]
    else:
        # If there is no saved original platforms list, save the current one.
        current_platforms = settings["//command_line_option:platforms"]
        original_platforms = [str(label) for label in current_platforms]

    return {
        "//:original_platforms": original_platforms,
        "//command_line_option:platforms": [str(attr.platform)],
    }

set_platform_transition = transition(
    inputs = [
        "//command_line_option:platforms",
        "//:original_platforms",
    ],
    outputs = [
        "//command_line_option:platforms",
        "//:original_platforms",
    ],
    implementation = _set_platform_transition_impl,
)

def _set_platform_filegroup_impl(ctx):
    return [DefaultInfo(files = depset(ctx.files.srcs))]

set_platform_filegroup = rule(
    implementation = _set_platform_filegroup_impl,
    attrs = {
        # The platform to use when building the runtimes.
        "platform": attr.label(mandatory = False),

        # Mark that our dependencies are built through a transition.
        "srcs": attr.label_list(mandatory = True, cfg = set_platform_transition),

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
        runtimes,
        build_stage = None,
        base_stage = None,
        tags = []):
    """Create a toolchain suite that uses the local Clang/LLVM install.

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
        runtimes: A list of runtimes to include in the toolchain.
        tags: Tags to apply to the toolchain.
    """

    def _platform_name(os, cpu, name_suffix = ""):
        return "{}{}_{}_{}_platform".format(name, name_suffix, os, cpu)

    # Define platforms for each supported OS/CPU pair.
    for os, cpus in platforms.items():
        for cpu in cpus:
            constraint_values = [
                "@platforms//os:" + os,
                "@platforms//cpu:" + cpu,
            ]
            if base_stage:
                native.platform(
                    name = _platform_name(os, cpu, "_base"),
                    constraint_values = constraint_values + [base_stage],
                )
            if build_stage:
                constraint_values.append(build_stage)
            native.platform(
                name = _platform_name(os, cpu),
                constraint_values = constraint_values,
            )
            native.platform(
                name = _platform_name(os, cpu, "_runtimes"),
                constraint_values = constraint_values + [":is_runtimes_build"],
            )

    base_platform_select = None
    if base_stage:
        base_platform_select = select({
            ":is_{}_{}".format(os, cpu): ":" + _platform_name(os, cpu, "_base")
            for os, cpus in platforms.items()
            for cpu in cpus
        })

    runtimes_platform_select = select({
        ":is_{}_{}".format(os, cpu): ":" + _platform_name(os, cpu, "_runtimes")
        for os, cpus in platforms.items()
        for cpu in cpus
    })

    set_platform_filegroup(
        name = name + "_base_files",
        srcs = base_files,
        platform = base_platform_select,
        tags = tags,
    )

    set_platform_filegroup(
        name = name + "_runtimes_compile_files",
        srcs = [":" + name + "_base_files"] + clang_hdrs,
        platform = base_platform_select,
        tags = tags,
    )

    set_platform_filegroup(
        name = name + "_compile_files",
        srcs = [":" + name + "_base_files"] + all_hdrs,
        platform = base_platform_select,
        tags = tags,
    )

    carbon_cc_toolchain_config(
        name = name + "_runtimes_toolchain_config",
        identifier_prefix = name + "_runtimes",
        target_cpu = select({
            # Note that we need to select on both OS and CPU so that we end up
            # spelling the CPU in the correct OS-specific ways.
            ":is_{}_{}".format(os, cpu): cpu
            for os, cpus in platforms.items()
            for cpu in cpus
        }),
        target_os = select({
            "@platforms//os:" + os: os
            for os in platforms.keys()
        }),
        bins = ":" + name + "_base_files",
        tags = tags,
    )

    cc_toolchain(
        name = name + "_runtimes_cc_toolchain",
        all_files = ":" + name + "_runtimes_compile_files",
        ar_files = ":" + name + "_base_files",
        as_files = ":" + name + "_runtimes_compile_files",
        compiler_files = ":" + name + "_runtimes_compile_files",
        dwp_files = ":" + name + "_base_files",
        linker_files = ":" + name + "_base_files",
        objcopy_files = ":" + name + "_base_files",
        strip_files = ":" + name + "_base_files",
        toolchain_config = ":" + name + "_runtimes_toolchain_config",
        toolchain_identifier = select({
            ":is_{}_{}".format(os, cpu): _platform_name(os, cpu, "_runtimes")
            for os, cpus in platforms.items()
            for cpu in cpus
        }),
        tags = tags,
    )

    native.toolchain(
        name = name + "_runtimes_toolchain",
        target_compatible_with = [":is_runtimes_build"],
        toolchain = ":" + name + "_runtimes_cc_toolchain",
        toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
        tags = tags,
    )

    set_platform_filegroup(
        name = name + "_runtimes",
        srcs = [runtimes],
        platform = runtimes_platform_select,
        tags = tags,
    )

    carbon_cc_toolchain_config(
        name = name + "_toolchain_config",
        identifier_prefix = name,
        target_cpu = select({
            # Note that we need to select on both OS and CPU so that we end up
            # spelling the CPU in the correct OS-specific ways.
            ":is_{}_{}".format(os, cpu): cpu
            for os, cpus in platforms.items()
            for cpu in cpus
        }),
        target_os = select({
            "@platforms//os:" + os: os
            for os in platforms.keys()
        }),
        runtimes = ":" + name + "_runtimes",
        bins = ":" + name + "_base_files",
        tags = tags,
    )

    native.filegroup(
        name = name + "_linker_files",
        srcs = [
            ":" + name + "_base_files",
            ":" + name + "_runtimes",
        ],
        tags = tags,
    )

    native.filegroup(
        name = name + "_all_files",
        srcs = [
            ":" + name + "_compile_files",
            ":" + name + "_linker_files",
        ],
        tags = tags,
    )

    cc_toolchain(
        name = name + "_cc_toolchain",
        all_files = ":" + name + "_all_files",
        ar_files = ":" + name + "_base_files",
        as_files = ":" + name + "_compile_files",
        compiler_files = ":" + name + "_compile_files",
        dwp_files = ":" + name + "_linker_files",
        linker_files = ":" + name + "_linker_files",
        objcopy_files = ":" + name + "_base_files",
        strip_files = ":" + name + "_base_files",
        toolchain_config = ":" + name + "_toolchain_config",
        toolchain_identifier = select({
            ":is_{}_{}".format(os, cpu): _platform_name(os, cpu)
            for os, cpus in platforms.items()
            for cpu in cpus
        }),
        tags = tags,
    )

    native.toolchain(
        name = name + "_toolchain",
        target_compatible_with = [build_stage] if build_stage else [],
        toolchain = ":" + name + "_cc_toolchain",
        toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
        tags = tags,
    )
