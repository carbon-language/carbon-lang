# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Starlark cc_toolchain configuration rules for using the Carbon toolchain"""

load(
    "@carbon_toolchain_config//:carbon_detected_variables.bzl",
    "clang_include_dirs",
    "clang_sysroot",
)
load(
    "@rules_cc//cc:defs.bzl",
    "CcToolchainConfigInfo",
    "cc_toolchain",
)
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load(":cc_toolchain_features.bzl", "clang_cc_toolchain_features")
load(
    ":cc_toolchain_tools.bzl",
    "llvm_action_configs",
    "llvm_tool_paths",
)

def _impl(ctx):
    # Hard code the the repository-relative path of the LLVM (and Clang)
    # binaries as it is a fixed aspect of the install structure.
    llvm_bindir = "llvm/bin"

    # Only use a sysroot if a non-trivial one is set in Carbon's config.
    builtin_sysroot = None
    if clang_sysroot != "None" and clang_sysroot != "/":
        builtin_sysroot = clang_sysroot

    identifier = "carbon-toolchain-{0}-{1}".format(ctx.attr.target_cpu, ctx.attr.target_os)
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = clang_cc_toolchain_features(
            target_os = ctx.attr.target_os,
            target_cpu = ctx.attr.target_cpu,
        ),
        action_configs = llvm_action_configs(llvm_bindir),
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
    implementation = _impl,
    attrs = {
        "target_cpu": attr.string(mandatory = True),
        "target_os": attr.string(mandatory = True),
    },
    provides = [CcToolchainConfigInfo],
)

def carbon_cc_toolchain_suite(name, configs):
    """Create a toolchain suite that uses the local Clang/LLVM install.

    Args:
        name: The name of the toolchain suite to produce.
        configs: An array of (os, cpu) pairs to support in the toolchain.
    """

    native.filegroup(
        name = name + "_files",
        srcs = native.glob([
            "**",
        ]),
    )

    # Create the individual local toolchains for each CPU.
    for (os, cpu) in configs:
        config_name = "{0}_{1}_{2}".format(name, os, cpu)
        carbon_cc_toolchain_config(
            name = config_name + "_config",
            target_cpu = cpu,
            target_os = os,
        )
        cc_toolchain(
            name = config_name + "_tools",
            all_files = ":" + name + "_files",
            ar_files = ":" + name + "_files",
            as_files = ":" + name + "_files",
            compiler_files = ":" + name + "_files",
            dwp_files = ":" + name + "_files",
            linker_files = ":" + name + "_files",
            objcopy_files = ":" + name + "_files",
            strip_files = ":" + name + "_files",
            supports_param_files = 1,
            toolchain_config = ":" + config_name + "_config",
            toolchain_identifier = config_name,
        )
        compatible_with = ["@platforms//cpu:" + cpu, "@platforms//os:" + os]
        native.toolchain(
            name = config_name,
            exec_compatible_with = compatible_with,
            target_compatible_with = compatible_with,
            toolchain = config_name + "_tools",
            toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
        )
