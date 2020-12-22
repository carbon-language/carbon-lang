# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Aspect to run `clang-format` over relevant sources.

The `clang-format` tool is taken from the `CcToolchainProvider`.
"""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load(
    "@bazel_tools//tools/build_defs/cc:action_names.bzl",
    "CPP_COMPILE_ACTION_NAME",
    "C_COMPILE_ACTION_NAME",
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

    sources = []
    if hasattr(ctx.rule.attr, 'srcs'):
        sources += ctx.rule.attr.srcs
    if hasattr(ctx.rule.attr, 'hdrs'):
        sources += ctx.rule.attr.hdrs
    if hasattr(ctx.rule.attr, 'textual_hdrs'):
        sources += ctx.rule.attr.textual_hdrs

    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    compiler = str(
        cc_common.get_tool_for_action(
            feature_configuration = feature_configuration,
            action_name = C_COMPILE_ACTION_NAME,
        ),
    )
    
    # FIXME: This is a pretty gross hack that relies on a particular spelling
    # of both the compiler and the format tool. Ideally, we'd extract this from
    # the `tool_paths` of the toolchain, but it isn't clear how to actually
    # access that structure.
    clang_format = compiler + "-format";

    for src in sources:
        ctx.actions.run(
            inputs = src.files,
            outputs = [],
            arguments = ["--dry-run", "-Werror"],
            progress_message = "Checking `clang-format` of `%s`" % src,
            executable = clang_format,
        )
    return []

clang_format_aspect = aspect(
    implementation = _clang_format_impl,
    attr_aspects = [],
    attrs = {
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
    },
    fragments = ["cpp"],
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
)
