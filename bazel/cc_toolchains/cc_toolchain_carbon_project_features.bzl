# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Defines `cc_toolchain_config` features specific to the Carbon project."""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES", "ACTION_NAME_GROUPS")
load(
    "@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "with_feature_set",
)
load(
    ":cc_toolchain_actions.bzl",
    "preprocessor_compile_actions",
)

# An enabled feature that requires the `fastbuild` compilation. This is used
# to toggle general features on by default, while allowing them to be
# directly enabled and disabled more generally as desired.
carbon_project_fastbuild_feature = feature(
    name = "enable_in_fastbuild",
    enabled = True,
    requires = [feature_set(["fastbuild"])],
    implies = [
        "minimal_optimization_flags",
        "minimal_debug_info_flags",
        "preserve_call_stacks",
    ],
)

def carbon_project_features(cache_key):
    return [carbon_project_fastbuild_feature, feature(
        name = "project_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = ACTION_NAME_GROUPS.all_cc_compile_actions,
                flag_groups = [flag_group(flags = [
                    # Don't warn on external code as we can't
                    # necessarily patch it easily. Note that these have
                    # to be initial directories in the `#include` line.
                    "--system-header-prefix=absl/",
                    "--system-header-prefix=benchmark/",
                    "--system-header-prefix=boost/",
                    "--system-header-prefix=clang-tools-extra/",
                    "--system-header-prefix=clang/",
                    "--system-header-prefix=gmock/",
                    "--system-header-prefix=gtest/",
                    "--system-header-prefix=libfuzzer/",
                    "--system-header-prefix=llvm/",
                    "--system-header-prefix=re2/",
                    "--system-header-prefix=tools/cpp/",
                    "--system-header-prefix=tree_sitter/",
                ])],
            ),
            flag_set(
                actions = preprocessor_compile_actions,
                flag_groups = [flag_group(flags = [
                    # Pass a cache key as a `-D` flag to avoid unintended Bazel
                    # cache hits when the underlying toolchain changes.
                    # TODO: We should consider replacing this by causing changes
                    # to the installed toolchain to more reliably end up as part
                    # of the action digest.
                    "-DBAZEL_COMPILE_CACHE_KEY=\"%s\"" % cache_key,
                ])],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [flag_group(flags = ["-DHAVE_MALLCTL"])],
                with_features = [with_feature_set(["freebsd_target"])],
            ),
        ],
    )]
