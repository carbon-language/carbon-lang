# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Starlark cc_toolchain configuration rules for using the Carbon toolchain"""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES", "ACTION_NAME_GROUPS")
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
)
load("cc_toolchain_carbon_project_features.bzl", "carbon_project_features")
load("cc_toolchain_features.bzl", "clang_cc_toolchain_features")
load(
    ":cc_toolchain_tools.bzl",
    "llvm_tool_paths",
)

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
        for name in ACTION_NAME_GROUPS.all_cpp_compile_actions
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
        for name in ACTION_NAME_GROUPS.all_cc_link_actions
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
    llvm_bindir = "llvm/bin"
    clang_bindir = llvm_bindir
    tools = struct(
        carbon_busybox = tool(path = "carbon-busybox"),
        clang = tool(path = clang_bindir + "/clang"),
        clangpp = tool(path = clang_bindir + "/clang++"),
        llvm_ar = tool(path = llvm_bindir + "/llvm-ar"),
        llvm_strip = tool(path = llvm_bindir + "/llvm-strip"),
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
        if not all([carbon_busybox, clang, clangpp, llvm_ar, llvm_strip]):
            fail("Missing required tool in bins: {0}".format(ctx.attr.bins))
        llvm_bindir = llvm_ar.dirname
        clang_bindir = clang.dirname
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

        # Pass in our tool paths to expose Make variables like $(NM) and
        # $(OBJCOPY).
        tool_paths = llvm_tool_paths(llvm_bindir, clang_bindir),
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

def _runtimes_transition_impl(_, attr):
    return {
        "//:runtimes_build": True,
    }

_runtimes_transition = transition(
    inputs = [],
    outputs = [
        "//:runtimes_build",
    ],
    implementation = _runtimes_transition_impl,
)

def _filegroup_with_runtimes_build_impl(ctx):
    return [DefaultInfo(files = depset(ctx.files.srcs))]

filegroup_with_runtimes_build = rule(
    implementation = _filegroup_with_runtimes_build_impl,
    attrs = {
        "srcs": attr.label_list(mandatory = True, cfg = _runtimes_transition),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
    doc = "A filegroup whose sources are built with or without runtimes building enabled.",
)

def carbon_cc_toolchain(
        name,
        platforms,
        base_files_target,
        runtimes_compile_files_target,
        compile_files_target,
        runtimes_target,
        extra_toolchain_settings = [],
        tags = []):
    """Create a Carbon `cc_toolchain` for the current target.

    This macro constructs the configuration and toolchain rules for a baseline
    Carbon toolchain, including building its own runtimes on demand.

    Args:
        name: The base name for the toolchain targets.
        platforms: Supported platforms.
        base_files_target: Target for base files.
        runtimes_compile_files_target: Target for runtimes compile files.
        compile_files_target: Target for compile files.
        runtimes_target: Target for runtimes.
        extra_toolchain_settings: Extra toolchain settings.
        tags: Tags to apply to the toolchain.
    """
    impl_tags = tags if "manual" in tags else tags + ["manual"]

    carbon_cc_toolchain_config(
        name = "{}_runtimes_toolchain_config".format(name),
        identifier_prefix = "{}_runtimes".format(name),
        target_cpu = select({
            ":is_{}_{}".format(os, cpu): cpu
            for os, cpus in platforms.items()
            for cpu in cpus
        }),
        target_os = select({
            "@platforms//os:{}".format(os): os
            for os in platforms.keys()
        }),
        bins = base_files_target,
        tags = impl_tags,
    )

    cc_toolchain(
        name = "{}_runtimes_cc_toolchain".format(name),
        all_files = runtimes_compile_files_target,
        ar_files = base_files_target,
        as_files = runtimes_compile_files_target,
        compiler_files = runtimes_compile_files_target,
        dwp_files = base_files_target,
        linker_files = base_files_target,
        objcopy_files = base_files_target,
        strip_files = base_files_target,
        toolchain_config = ":{}_runtimes_toolchain_config".format(name),
        toolchain_identifier = select({
            ":is_{}_{}".format(os, cpu): "{}_{}_{}_runtimes_toolchain".format(name, os, cpu)
            for os, cpus in platforms.items()
            for cpu in cpus
        }),
        tags = impl_tags,
    )

    native.toolchain(
        name = "{}_runtimes_toolchain".format(name),
        target_settings = [":is_runtimes_build"] + extra_toolchain_settings,
        use_target_platform_constraints = True,
        toolchain = ":{}_runtimes_cc_toolchain".format(name),
        toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
        tags = tags,
    )

    carbon_cc_toolchain_config(
        name = "{}_toolchain_config".format(name),
        identifier_prefix = name,
        target_cpu = select({
            ":is_{}_{}".format(os, cpu): cpu
            for os, cpus in platforms.items()
            for cpu in cpus
        }),
        target_os = select({
            "@platforms//os:{}".format(os): os
            for os in platforms.keys()
        }),
        runtimes = runtimes_target,
        bins = base_files_target,
        tags = impl_tags,
    )

    native.filegroup(
        name = "{}_linker_files".format(name),
        srcs = [
            base_files_target,
            runtimes_target,
        ],
        tags = impl_tags,
    )

    native.filegroup(
        name = "{}_all_files".format(name),
        srcs = [
            compile_files_target,
            ":{}_linker_files".format(name),
        ],
        tags = impl_tags,
    )

    cc_toolchain(
        name = "{}_cc_toolchain".format(name),
        all_files = ":{}_all_files".format(name),
        ar_files = base_files_target,
        as_files = compile_files_target,
        compiler_files = compile_files_target,
        dwp_files = ":{}_linker_files".format(name),
        linker_files = ":{}_linker_files".format(name),
        objcopy_files = base_files_target,
        strip_files = base_files_target,
        toolchain_config = ":" + name + "_toolchain_config",
        toolchain_identifier = select({
            ":is_{}_{}".format(os, cpu): "{}_{}_{}_toolchain".format(name, os, cpu)
            for os, cpus in platforms.items()
            for cpu in cpus
        }),
        tags = impl_tags,
    )

    native.toolchain(
        name = name + "_toolchain",
        target_settings = [":not_runtimes_build"] + extra_toolchain_settings,
        use_target_platform_constraints = True,
        toolchain = ":" + name + "_cc_toolchain",
        toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
        tags = tags,
    )
