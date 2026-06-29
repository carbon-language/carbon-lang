# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provides rules for building Carbon files using the toolchain."""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _carbon_binary_impl(ctx):
    toolchain_driver = ctx.executable.internal_exec_toolchain_driver
    toolchain_data = ctx.files.internal_exec_toolchain_data
    prebuilt_runtimes = ctx.files.internal_exec_prebuilt_runtimes

    # If the exec driver isn't provided, that means we're trying to use a target
    # config toolchain, likely to avoid build overhead of two configs.
    if toolchain_driver == None:
        toolchain_driver = ctx.executable.internal_target_toolchain_driver
        toolchain_data = ctx.files.internal_target_toolchain_data
        prebuilt_runtimes = ctx.files.internal_target_prebuilt_runtimes

    # The extra link flags needed.
    link_flags = []

    # Pass any C++ flags from our dependencies onto Carbon.
    dep_flags = []
    dep_hdrs = []
    dep_api_files = []
    dep_link_inputs = []
    for dep in ctx.attr.deps:
        if CcInfo in dep:
            cc_info = dep[CcInfo]

            # TODO: We should reuse the feature-based flag generation in
            # bazel/cc_toolchains here.
            dep_flags += ["--clang-arg=-D{0}".format(define) for define in cc_info.compilation_context.defines.to_list()]
            dep_flags += ["--clang-arg=-I{0}".format(path) for path in cc_info.compilation_context.includes.to_list()]
            dep_flags += ["--clang-arg=-iquote{0}".format(path) for path in cc_info.compilation_context.quote_includes.to_list()]
            dep_flags += ["--clang-arg=-isystem{0}".format(path) for path in cc_info.compilation_context.system_includes.to_list()]
            dep_hdrs.append(cc_info.compilation_context.headers)
            for link_input in cc_info.linking_context.linker_inputs.to_list():
                link_flags += link_input.user_link_flags
                dep_link_inputs += link_input.additional_inputs
                for lib in link_input.libraries:
                    dep_link_inputs += [dep for dep in [lib.dynamic_library, lib.static_library] if dep]
                    dep_link_inputs += lib.objects
        if DefaultInfo in dep:
            dep_link_inputs += dep[DefaultInfo].files.to_list()
        if CarbonLibraryInfo in dep:
            carbon_info = dep[CarbonLibraryInfo]
            dep_link_inputs += carbon_info.objs.to_list()
            dep_api_files.append(carbon_info.api)

    # Add the dependencies' link flags and inputs to the link flags.
    link_flags += [dep.path for dep in dep_link_inputs]

    # Build object files for the prelude and for the binary itself.
    # TODO: Eventually the prelude should be build as a separate `carbon_library`.
    srcs_and_flags = [(ctx.files.srcs, dep_flags)]

    objs = []
    for (srcs, extra_flags) in srcs_and_flags:
        for src in srcs:
            # Build each source file. For now, we pass all sources to each compile
            # because we don't have visibility into dependencies and have no way to
            # specify multiple output files. Object code for each input is written
            # into the output file in turn, so the final carbon source file
            # specified ends up determining the contents of the object file.
            #
            # TODO: This is a hack; replace with something better once the toolchain
            # supports doing so.
            #
            # TODO: Switch to the `prefix` based rule similar to linking when
            # the prelude moves there.
            out = ctx.actions.declare_file("_objs/{0}/{1}o".format(
                ctx.label.name,
                src.short_path.removeprefix(ctx.label.package).removesuffix(src.extension),
            ))
            objs.append(out)
            srcs_reordered = dep_api_files + [s for s in srcs if s != src] + [src]
            ctx.actions.run(
                outputs = [out],
                inputs = depset(direct = srcs_reordered, transitive = dep_hdrs),
                executable = toolchain_driver,
                tools = depset(toolchain_data),
                arguments = ["compile", "--output=" + out.path, "--output-last-input-only"] +
                            [s.path for s in srcs_reordered] + extra_flags + ctx.attr.flags,
                mnemonic = "CarbonCompile",
                progress_message = "Compiling " + src.short_path,
            )

    # Add the Carbon object files to the link flags.
    link_flags += [o.path for o in objs]

    bin = ctx.actions.declare_file(ctx.label.name)

    # Get all link options from the toolchain and dependencies using standard pattern.
    cc_toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    variables = cc_common.create_link_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        is_using_linker = True,
        user_link_flags = link_flags + [
            # TODO: Remove once the sanitizer runtimes are available.
            "-fno-sanitize=all",
        ],
        output_file = bin.path,
    )
    full_link_flags = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_executable,
        variables = variables,
    )

    ctx.actions.run(
        outputs = [bin],
        inputs = objs + dep_link_inputs,
        executable = toolchain_driver,
        tools = depset(toolchain_data + prebuilt_runtimes),
        arguments = full_link_flags,
        mnemonic = "CarbonLink",
        progress_message = "Linking " + bin.short_path,
    )
    return [DefaultInfo(files = depset([bin]), executable = bin)]

CarbonLibraryInfo = provider(
    doc = "Contains information about a compiled Carbon library.",
    fields = {
        "api": "The api source file to provide to library consumers.",
        "objs": "A depset of one or more compiled library files, including impl and api.",
    },
)

def _carbon_library_impl(ctx):
    toolchain_driver = ctx.executable.internal_exec_toolchain_driver
    toolchain_data = ctx.files.internal_exec_toolchain_data

    # If the exec driver isn't provided, that means we're trying to use a target
    # config toolchain, likely to avoid build overhead of two configs.
    if toolchain_driver == None:
        toolchain_driver = ctx.executable.internal_target_toolchain_driver
        toolchain_data = ctx.files.internal_target_toolchain_data

    # Pass any C++ flags from our dependencies onto Carbon.
    dep_flags = []
    dep_hdrs = []
    dep_api_srcs = []
    for dep in ctx.attr.deps:
        if CcInfo in dep:
            cc_info = dep[CcInfo]

            # TODO: We should reuse the feature-based flag generation in
            # bazel/cc_toolchains here.
            dep_flags += ["--clang-arg=-D{0}".format(define) for define in cc_info.compilation_context.defines.to_list()]
            dep_flags += ["--clang-arg=-I{0}".format(path) for path in cc_info.compilation_context.includes.to_list()]
            dep_flags += ["--clang-arg=-iquote{0}".format(path) for path in cc_info.compilation_context.quote_includes.to_list()]
            dep_flags += ["--clang-arg=-isystem{0}".format(path) for path in cc_info.compilation_context.system_includes.to_list()]
            dep_hdrs.append(cc_info.compilation_context.headers)
        if CarbonLibraryInfo in dep:
            carbon_info = dep[CarbonLibraryInfo]
            dep_api_srcs.append(carbon_info.api)

    # Build object files for the library impls and api file
    srcs_and_flags = [(ctx.files.impls + ctx.files.api, dep_flags)]

    objs = []
    for (srcs, extra_flags) in srcs_and_flags:
        for src in srcs:
            # Build each source file. For now, we pass all sources to each compile
            # because we don't have visibility into dependencies and have no way to
            # specify multiple output files. Object code for each input is written
            # into the output file in turn, so the final carbon source file
            # specified ends up determining the contents of the object file.
            #
            # TODO: This is a hack; replace with something better once the toolchain
            # supports doing so.
            #
            # TODO: Switch to the `prefix` based rule similar to linking when
            # the prelude moves there.
            out = ctx.actions.declare_file("_objs/{0}/{1}o".format(
                ctx.label.name,
                src.short_path.removeprefix(ctx.label.package).removesuffix(src.extension),
            ))
            objs.append(out)
            srcs_reordered = dep_api_srcs + [s for s in srcs if s != src] + [src]
            ctx.actions.run(
                outputs = [out],
                inputs = depset(direct = srcs_reordered, transitive = dep_hdrs),
                executable = toolchain_driver,
                tools = depset(toolchain_data),
                arguments = ["compile", "--output=" + out.path, "--output-last-input-only"] +
                            [s.path for s in srcs_reordered] + extra_flags + ctx.attr.flags,
                mnemonic = "CarbonCompile",
                progress_message = "Compiling " + src.short_path,
            )

    return [CarbonLibraryInfo(api = ctx.files.api[0], objs = depset(objs))]

def _carbon_prelude_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)

    # TODO: find a less terrible way to figure this out
    toolchain_root = str(cc_toolchain.compiler_executable).removeprefix(cc_toolchain._crosstool_top_path + "/").removesuffix("toolchain/install/llvm/bin/clang++")
    carbon_busybox = toolchain_root + cc_toolchain._tool_paths["carbon-busybox"]
    srcs = [s for s in ctx.files.srcs if s.extension == "carbon"]
    objs = []
    for src in srcs:
        out = ctx.actions.declare_file("_objs/{0}/{1}o".format(
            ctx.label.name,
            src.short_path.removeprefix(ctx.label.package).removesuffix(src.extension),
        ))
        objs.append(out)
        srcs_reordered = [s for s in srcs if s != src] + [src]
        ctx.actions.run(
            outputs = [out],
            inputs = depset(direct = srcs_reordered),
            tools = depset(transitive = [cc_toolchain.all_files]),
            executable = carbon_busybox,
            arguments = ["compile", "--output=" + out.path, "--output-last-input-only", "--no-prelude-import"] +
                        [s.path for s in srcs_reordered] + ctx.attr.flags,
            mnemonic = "CarbonPrelude",
            progress_message = "Precompiling prelude file " + src.short_path,
        )

    return DefaultInfo(files = depset(objs))

_carbon_binary_internal = rule(
    implementation = _carbon_binary_impl,
    attrs = {
        "deps": attr.label_list(allow_files = True, providers = [[CcInfo], [DefaultInfo], [CarbonLibraryInfo]]),
        "flags": attr.string_list(),

        # The exec config toolchain attributes. These will be `None` when using
        # the target config and populated when using the exec config. We have to
        # use duplicate attributes here and below to have different `cfg`
        # settings, as that isn't `select`-able, and we'll use `select`s when
        # populating these.
        "internal_exec_prebuilt_runtimes": attr.label(
            cfg = "exec",
        ),
        "internal_exec_toolchain_data": attr.label(
            cfg = "exec",
        ),
        "internal_exec_toolchain_driver": attr.label(
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),

        # The target config toolchain attributes. These will be 'None' when
        # using the exec config and populated when using the target config. We
        # have to use duplicate attributes here and below to have different
        # `cfg` settings, as that isn't `select`-able, and we'll use `select`s
        # when populating these.
        "internal_target_prebuilt_runtimes": attr.label(
            cfg = "target",
        ),
        "internal_target_toolchain_data": attr.label(
            cfg = "target",
        ),
        "internal_target_toolchain_driver": attr.label(
            allow_single_file = True,
            executable = True,
            cfg = "target",
        ),
        "prelude_srcs": attr.label_list(allow_files = [".carbon"]),
        "srcs": attr.label_list(allow_files = [".carbon"]),
        "_cc_toolchain": attr.label(default = "//toolchain/install:carbon_stage1_cc_toolchain"),
    },
    executable = True,
    fragments = ["cpp"],
)

_carbon_library_internal = rule(
    implementation = _carbon_library_impl,
    attrs = {
        "api": attr.label(allow_single_file = True),
        "deps": attr.label_list(allow_files = True),
        "flags": attr.string_list(),
        "impls": attr.label_list(allow_files = [".carbon"]),

        # The exec config toolchain attributes. These will be `None` when using
        # the target config and populated when using the exec config. We have to
        # use duplicate attributes here and below to have different `cfg`
        # settings, as that isn't `select`-able, and we'll use `select`s when
        # populating these.
        "internal_exec_prebuilt_runtimes": attr.label(
            cfg = "exec",
        ),
        "internal_exec_toolchain_data": attr.label(
            cfg = "exec",
        ),
        "internal_exec_toolchain_driver": attr.label(
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),

        # The target config toolchain attributes. These will be 'None' when
        # using the exec config and populated when using the target config. We
        # have to use duplicate attributes here and below to have different
        # `cfg` settings, as that isn't `select`-able, and we'll use `select`s
        # when populating these.
        "internal_target_prebuilt_runtimes": attr.label(
            cfg = "target",
        ),
        "internal_target_toolchain_data": attr.label(
            cfg = "target",
        ),
        "internal_target_toolchain_driver": attr.label(
            allow_single_file = True,
            executable = True,
            cfg = "target",
        ),
        "prelude_srcs": attr.label_list(allow_files = [".carbon"]),
        "_cc_toolchain": attr.label(default = "//toolchain/install:carbon_stage1_cc_toolchain"),
    },
    executable = False,
    fragments = ["cpp"],
)

carbon_prelude = rule(
    implementation = _carbon_prelude_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = [".carbon"]),
        "flags": attr.string_list(),
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)

# We synthesize two sets of attributes from mirrored `select`s here
# because we want to select on an internal property of these attributes
# but that isn't `select`-able. Instead, we have both attributes and
# `select` which one we use.
_select_internal_exec_toolchain_driver = select({
    Label("//bazel/carbon_rules:use_target_config_carbon_rules_config"): None,
    "//conditions:default": Label("//toolchain/install:carbon-busybox"),
})
_select_internal_exec_toolchain_data = select({
    Label("//bazel/carbon_rules:use_target_config_carbon_rules_config"): None,
    "//conditions:default": Label("//toolchain/install:install_data"),
})
_select_internal_exec_prebuilt_runtimes = select({
    Label("//bazel/carbon_rules:use_target_config_carbon_rules_config"): None,
    "//conditions:default": Label("//toolchain/install:built_runtimes"),
})
_select_internal_target_toolchain_driver = select({
    Label(
        "//bazel/carbon_rules:use_target_config_carbon_rules_config",
    ): Label("//toolchain/install:carbon-busybox"),
    "//conditions:default": None,
})
_select_internal_target_toolchain_data = select({
    Label(
        "//bazel/carbon_rules:use_target_config_carbon_rules_config",
    ): Label("//toolchain/install:install_data"),
    "//conditions:default": None,
})
_select_internal_target_prebuilt_runtimes = select({
    Label(
        "//bazel/carbon_rules:use_target_config_carbon_rules_config",
    ): Label("//toolchain/install:built_runtimes"),
    "//conditions:default": None,
})

def carbon_binary(name, srcs, deps = [], flags = [], tags = []):
    """Compiles a Carbon binary.

    Args:
      name: The name of the build target.
      srcs: List of Carbon source files to compile.
      deps: List of dependencies.
      flags: Extra flags to pass to the Carbon compile command.
      tags: Tags to apply to the rule.
    """
    _carbon_binary_internal(
        name = name,
        srcs = srcs,
        prelude_srcs = [Label("//core:prelude_files")],
        deps = deps + [Label("//core:io"), Label("//core:range")],
        flags = flags,
        tags = tags,
        internal_exec_toolchain_driver = _select_internal_exec_toolchain_driver,
        internal_exec_toolchain_data = _select_internal_exec_toolchain_data,
        internal_exec_prebuilt_runtimes = _select_internal_exec_prebuilt_runtimes,
        internal_target_toolchain_driver = _select_internal_target_toolchain_driver,
        internal_target_toolchain_data = _select_internal_target_toolchain_data,
        internal_target_prebuilt_runtimes = _select_internal_target_prebuilt_runtimes,
    )

def carbon_library(name, api, impls = [], deps = [], flags = [], tags = [], visibility = []):
    """Compiles a Carbon library.

    Args:
      name: The name of the build target.
      api: Name of a single api file.
      impls: List of zero or more implementation files.
      deps: List of dependencies.
      flags: Extra flags to pass to the Carbon compile command.
      tags: Tags to apply to the rule.
      visibility: Visibility rules for the library.
    """
    _carbon_library_internal(
        name = name,
        api = api,
        impls = impls,
        prelude_srcs = [Label("//core:prelude_files")],
        deps = deps,
        flags = flags,
        tags = tags,
        visibility = visibility,
        internal_exec_toolchain_driver = _select_internal_exec_toolchain_driver,
        internal_exec_toolchain_data = _select_internal_exec_toolchain_data,
        internal_exec_prebuilt_runtimes = _select_internal_exec_prebuilt_runtimes,
        internal_target_toolchain_driver = _select_internal_target_toolchain_driver,
        internal_target_toolchain_data = _select_internal_target_toolchain_data,
        internal_target_prebuilt_runtimes = _select_internal_target_prebuilt_runtimes,
    )
