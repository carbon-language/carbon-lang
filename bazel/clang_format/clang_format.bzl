# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Aspect to run `clang-format` over relevant sources.

The `clang-format` tool is taken from the `CcToolchainProvider`.
"""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load(
    "@bazel_cc_toolchain//:extra_action_names.bzl",
    "EXTRA_ACTION_NAMES",
)

_cc_rules = [
    "cc_library",
    "cc_binary",
    "cc_test",
]

def _clang_format_impl(target, ctx):
    # Only check `clang-format` on sources of C/C++ rules.
    if ctx.rule.kind not in _cc_rules:
        return []

    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    clang_format = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = EXTRA_ACTION_NAMES.clang_format,
    )

    source_targets = []
    if hasattr(ctx.rule.attr, "srcs"):
        source_targets += ctx.rule.attr.srcs
    if hasattr(ctx.rule.attr, "hdrs"):
        source_targets += ctx.rule.attr.hdrs
    if hasattr(ctx.rule.attr, "textual_hdrs"):
        source_targets += ctx.rule.attr.textual_hdrs

    output_tree = ctx.label.name + ".clang_format_validation/"

    sources = [
        file
        for target in source_targets
        for file in target.files.to_list()
        # Filter out generated files.
        # TODO: Eventually, it would be nice to format them instead.
        if not (file.basename.endswith(".tab.cpp") or
                file.basename.endswith(".tab.h") or
                file.basename.endswith(".yy.cpp"))
    ]
    outputs = [
        ctx.actions.declare_file(output_tree + src.path + ".validation")
        for src in sources
    ]

    for src, out in zip(sources, outputs):
        ctx.actions.run(
            inputs = depset(
                direct = [src],
                transitive = [ctx.attr._clang_format_config.files],
            ),
            outputs = [out],
            arguments = [
                out.path,
                clang_format,
                "--dry-run",
                "-Werror",
                src.path,
            ],
            progress_message = "Checking `clang-format` of `%s`" % src,
            executable = ctx.executable._clang_format_runner,
        )
    return [
        OutputGroupInfo(_validation = depset(outputs)),
    ]

clang_format_aspect = aspect(
    implementation = _clang_format_impl,
    attr_aspects = [],
    attrs = {
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
        "_clang_format_config": attr.label(
            default = Label("//:clang_format_config"),
        ),
        "_clang_format_runner": attr.label(
            default = Label("//bazel/clang_format:clang_format_runner.sh"),
            executable = True,
            allow_single_file = True,
            cfg = "exec",
        ),
    },
    fragments = ["cpp"],
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
)
