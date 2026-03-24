# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Starlark cc_toolchain configuration rules for using the Carbon toolchain"""

load(
    "@carbon_toolchain_config//:carbon_detected_variables.bzl",
    "clang_include_dirs",
    "clang_sysroot",
)
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
load("//bazel:runtimes_build_vars.bzl", "llvm_version_major")
load(
    ":cc_toolchain_actions.bzl",
    "all_c_compile_actions",
    "all_cpp_compile_actions",
    "all_link_actions",
)
load(":cc_toolchain_features.bzl", "clang_cc_toolchain_features")
load(
    ":cc_toolchain_tools.bzl",
    "llvm_tool_paths",
)

def _make_action_configs(runtimes_path = None):
    runtimes_flag = "--no-build-runtimes"
    if runtimes_path:
        runtimes_flag = "--prebuilt-runtimes={0}".format(runtimes_path)

    return [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tool(path = "llvm/bin/clang")],
        )
        for name in all_c_compile_actions
    ] + [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tool(path = "llvm/bin/clang++")],
        )
        for name in all_cpp_compile_actions
    ] + [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tool(path = "carbon-busybox")],
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
            tools = [tool(path = "llvm/bin/llvm-ar")],
        )
        for name in [ACTION_NAMES.cpp_link_static_library]
    ] + [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tool(path = "llvm/bin/llvm-strip")],
        )
        for name in [ACTION_NAMES.strip]
    ]

def _carbon_cc_toolchain_config_impl(ctx):
    # Hard code the the repository-relative path of the LLVM (and Clang)
    # binaries as it is a fixed aspect of the install structure.
    llvm_bindir = "llvm/bin"

    # Only use a sysroot if a non-trivial one is set in Carbon's config.
    builtin_sysroot = None
    if clang_sysroot != "None" and clang_sysroot != "/":
        builtin_sysroot = clang_sysroot

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

    identifier = "carbon-toolchain-{0}-{1}".format(
        ctx.attr.target_cpu,
        ctx.attr.target_os,
    )
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = clang_cc_toolchain_features(
            target_os = ctx.attr.target_os,
            target_cpu = ctx.attr.target_cpu,
        ),
        action_configs = _make_action_configs(runtimes_path),
        cxx_builtin_include_directories = clang_include_dirs,
        builtin_sysroot = builtin_sysroot,

        # This configuration only supports local non-cross builds so derive
        # everything from the target CPU selected.
        toolchain_identifier = identifier,

        # This is used to expose a "flag" that `config_setting` rules can use to
        # determine if the compiler is Clang.
        compiler = "clang",

        # We do have to pass in our tool paths.
        tool_paths = llvm_tool_paths(llvm_bindir),
    )

carbon_cc_toolchain_config = rule(
    implementation = _carbon_cc_toolchain_config_impl,
    attrs = {
        "runtimes": attr.label(mandatory = False),
        "target_cpu": attr.string(mandatory = True),
        "target_os": attr.string(mandatory = True),
    },
    provides = [CcToolchainConfigInfo],
)

def _runtimes_transition_impl(_, attr):
    # Adjust the platform to the runtimes platform across the transition.
    return {"//command_line_option:platforms": [str(attr.runtimes_platform)]}

runtimes_transition = transition(
    inputs = [],
    outputs = ["//command_line_option:platforms"],
    implementation = _runtimes_transition_impl,
)

def _runtimes_filegroup_impl(ctx):
    return [DefaultInfo(files = depset(ctx.files.srcs))]

runtimes_filegroup = rule(
    implementation = _runtimes_filegroup_impl,
    attrs = {
        # The platform to use when building the runtimes.
        "runtimes_platform": attr.label(mandatory = True),

        # Mark that our dependencies are built through a transition.
        "srcs": attr.label_list(mandatory = True, cfg = runtimes_transition),

        # Enable transitions in this rule.
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)

def carbon_cc_toolchain_suite(name, platforms):
    """Create a toolchain suite that uses the local Clang/LLVM install.

    Args:
        name: The name of the toolchain suite to produce.
        platforms: An array of (os, cpu) pairs to support in the toolchain.
    """

    # Our base filegroup for the toolchain just includes the busybox and the
    # LLVM symlinks. Importantly, this _doesn't_ include the install digest that
    # would cause cache misses as-if every file in the toolchain were an input
    # to every action.
    native.filegroup(
        name = name + "_base_files",
        srcs = ["carbon-busybox", "carbon_install.txt"] + native.glob([
            "llvm/bin/*",
        ]),
    )

    # We also need a compile-specific filegroup for use when _building_
    # runtimes. This needs to include the dedicated Clang headers, but we
    # require the runtimes themselves to provide the relevant headers and
    # dependencies.
    native.filegroup(
        name = name + "_runtimes_compile_files",
        srcs = [
            ":" + name + "_base_files",
            "//llvm/lib/clang/{0}:clang_hdrs".format(llvm_version_major),
        ],
    )

    # We can build a single common compile filegroup regardless of CPU and OS.
    # This needs to include all the headers of the runtimes, but those are
    # architecture independent.
    native.filegroup(
        name = name + "_compile_files",
        srcs = [
            ":" + name + "_runtimes_compile_files",
            "//runtimes:libunwind_hdrs",
            "//runtimes:libcxx_hdrs",
        ],
    )

    # Create the actual toolchains for each OS and CPU.
    for (os, cpu) in platforms:
        platform_name = "{0}_{1}_{2}".format(name, os, cpu)
        platform_constraints = [
            "@platforms//os:" + os,
            "@platforms//cpu:" + cpu,
        ]

        # First, configure a platform and toolchain for building runtimes. This
        # toolchain will only have the Clang headers and no runtime libraries
        # included.
        native.platform(
            name = platform_name + "_runtimes_platform",
            constraint_values = [":is_runtimes_build"] + platform_constraints,
        )
        carbon_cc_toolchain_config(
            name = platform_name + "_runtimes_toolchain_config",
            target_cpu = cpu,
            target_os = os,
        )
        cc_toolchain(
            name = platform_name + "_runtimes_cc_toolchain",
            all_files = ":" + name + "_runtimes_compile_files",
            ar_files = ":" + name + "_base_files",
            as_files = ":" + name + "_runtimes_compile_files",
            compiler_files = ":" + name + "_runtimes_compile_files",
            dwp_files = ":" + name + "_base_files",
            linker_files = ":" + name + "_base_files",
            objcopy_files = ":" + name + "_base_files",
            strip_files = ":" + name + "_base_files",
            toolchain_config = ":" + platform_name + "_runtimes_toolchain_config",
            toolchain_identifier = platform_name + "_runtimes",
        )
        native.toolchain(
            name = platform_name + "_runtimes_toolchain",
            exec_compatible_with = platform_constraints,
            target_compatible_with = [":is_runtimes_build"] + platform_constraints,
            toolchain = platform_name + "_runtimes_cc_toolchain",
            toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
        )

        # Now we can use the runtimes platform to build the runtimes on-demand.
        runtimes_filegroup(
            name = platform_name + "_runtimes",
            srcs = ["//runtimes:carbon_runtimes"],
            runtimes_platform = ":" + platform_name + "_runtimes_platform",
        )

        # Finally, we can build the main platform and toolchain.
        native.platform(
            name = platform_name + "_platform",
            constraint_values = platform_constraints,
        )

        # Build the main config with runtimes passed in. This allows the
        # configuration to use these built runtimes where needed.
        carbon_cc_toolchain_config(
            name = platform_name + "_toolchain_config",
            target_cpu = cpu,
            target_os = os,
            runtimes = ":" + platform_name + "_runtimes",
        )

        # We also include the runtimes in the linker files. We have to do this
        # in addition to the config parameter as the config can't carry the
        # actual dependency.
        native.filegroup(
            name = platform_name + "_linker_files",
            srcs = [
                ":" + name + "_base_files",
                ":" + platform_name + "_runtimes",
            ],
        )
        native.filegroup(
            name = platform_name + "_all_files",
            srcs = [
                ":" + name + "_compile_files",
                ":" + platform_name + "_linker_files",
            ],
        )
        cc_toolchain(
            name = platform_name + "_cc_toolchain",
            all_files = ":" + platform_name + "_all_files",
            ar_files = ":" + name + "_base_files",
            as_files = ":" + name + "_compile_files",
            compiler_files = ":" + name + "_compile_files",
            dwp_files = ":" + platform_name + "_linker_files",
            linker_files = ":" + platform_name + "_linker_files",
            objcopy_files = ":" + name + "_base_files",
            strip_files = ":" + name + "_base_files",
            toolchain_config = ":" + platform_name + "_toolchain_config",
            toolchain_identifier = platform_name,
        )
        native.toolchain(
            name = platform_name + "_toolchain",
            exec_compatible_with = platform_constraints,
            target_compatible_with = platform_constraints,
            toolchain = platform_name + "_cc_toolchain",
            toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
        )
