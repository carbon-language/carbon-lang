# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provides rules for building Carbon files using the toolchain."""

def _carbon_binary_impl(ctx):
    toolchain_driver = ctx.executable.internal_exec_toolchain_driver
    toolchain_data = ctx.files.internal_exec_toolchain_data

    # If the exec driver isn't provided, that means we're trying to use a target
    # config toolchain, likely to avoid build overhead of two configs.
    if toolchain_driver == None:
        toolchain_driver = ctx.executable.internal_target_toolchain_driver
        toolchain_data = ctx.files.internal_target_toolchain_data

    objs = []
    for src in ctx.files.srcs:
        # Build each source file. For now, we pass all sources to each compile
        # because we don't have visibility into dependencies and have no way to
        # specify multiple output files. Object code for each input is written
        # into the output file in turn, so the final carbon source file
        # specified ends up determining the contents of the object file.
        #
        # TODO: This is a hack; replace with something better once the toolchain
        # supports doing so.
        #
        # TODO: Switch to the `prefix_root` based rule similar to linking when
        # the prelude moves there.
        out = ctx.actions.declare_file("_objs/{0}/{1}o".format(
            ctx.label.name,
            src.short_path.removeprefix(ctx.label.package).removesuffix(src.extension),
        ))
        objs.append(out)
        srcs_reordered = [s for s in ctx.files.srcs if s != src] + [src]
        ctx.actions.run(
            outputs = [out],
            inputs = srcs_reordered,
            executable = toolchain_driver,
            tools = depset(toolchain_data),
            arguments = ["compile", "--output=" + out.path] + [s.path for s in srcs_reordered],
            mnemonic = "CarbonCompile",
            progress_message = "Compiling " + src.short_path,
        )

    # For now, we assume that the prelude doesn't produce any necessary object
    # code, and don't include the .o files for //core/prelude... in the final
    # linked binary.
    #
    # TODO: This will need to be revisited eventually.
    bin = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.run(
        outputs = [bin],
        inputs = objs,
        executable = toolchain_driver,
        tools = depset(toolchain_data),
        arguments = ["link", "--output=" + bin.path] + [o.path for o in objs],
        mnemonic = "CarbonLink",
        progress_message = "Linking " + bin.short_path,
    )
    return [DefaultInfo(files = depset([bin]))]

carbon_binary_internal = rule(
    implementation = _carbon_binary_impl,
    attrs = {
        # The exec config toolchain driver and data. These will be `None` when
        # using the target config and populated when using the exec config.
        "internal_exec_toolchain_data": attr.label(
            cfg = "exec",
        ),
        "internal_exec_toolchain_driver": attr.label(
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),

        # The target config toolchain driver and data. These will be 'None' when
        # using the exec config and populated when using the target config.
        "internal_target_toolchain_data": attr.label(
            cfg = "target",
        ),
        "internal_target_toolchain_driver": attr.label(
            allow_single_file = True,
            executable = True,
            cfg = "target",
        ),
        "srcs": attr.label_list(allow_files = [".carbon"]),
    },
)

def carbon_binary(name, srcs):
    """Compiles a Carbon binary.

    Args:
      name: The name of the build target.
      srcs: List of Carbon source files to compile.
    """
    carbon_binary_internal(
        name = name,
        srcs = srcs,
        internal_exec_toolchain_driver = select({
            "//bazel/carbon_rules:use_target_config_carbon_rules_config": None,
            "//conditions:default": "//toolchain/install:prefix_root/bin/carbon",
        }),
        internal_exec_toolchain_data = select({
            "//bazel/carbon_rules:use_target_config_carbon_rules_config": None,
            "//conditions:default": "//toolchain/install:install_data",
        }),
        internal_target_toolchain_driver = select({
            "//bazel/carbon_rules:use_target_config_carbon_rules_config": "//toolchain/install:prefix_root/bin/carbon",
            "//conditions:default": None,
        }),
        internal_target_toolchain_data = select({
            "//bazel/carbon_rules:use_target_config_carbon_rules_config": "//toolchain/install:install_data",
            "//conditions:default": None,
        }),
    )
