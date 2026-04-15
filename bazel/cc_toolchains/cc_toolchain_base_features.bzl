# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Definitions used for the base features of a `cc_toolchain_config`."""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load(
    "@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
)
load(
    ":cc_toolchain_actions.bzl",
    "all_compile_actions",
    "all_link_actions",
)

# Declare features that are used by Bazel to model specific build modes.
dbg_feature = feature(name = "dbg")
fastbuild_feature = feature(name = "fastbuild")
host_feature = feature(name = "host")
opt_feature = feature(name = "opt")

# Declare features that control enabling and disabling Bazel logic.
no_legacy_features_feature = feature(name = "no_legacy_features")
supports_pic_feature = feature(name = "supports_pic", enabled = True)

supports_dynamic_linker_feature = feature(
    name = "supports_dynamic_linker",
    enabled = True,
    requires = [feature_set(["linux_target"])],
)
supports_start_end_lib_feature = feature(
    name = "supports_start_end_lib",
    enabled = True,
    requires = [feature_set(["linux_target"])],
)

user_flags_feature = feature(
    name = "user_flags",
    enabled = True,
    flag_sets = [
        flag_set(
            actions = all_compile_actions,
            flag_groups = [flag_group(
                expand_if_available = "user_compile_flags",
                flags = ["%{user_compile_flags}"],
                iterate_over = "user_compile_flags",
            )],
        ),
        flag_set(
            actions = all_link_actions,
            flag_groups = [flag_group(
                expand_if_available = "user_link_flags",
                flags = ["%{user_link_flags}"],
                iterate_over = "user_link_flags",
            )],
        ),
    ],
)

# TODO: It's not clear this is the right location for these flags, and it is a
# little awkward.
output_flags_feature = feature(
    name = "output_flags",
    enabled = True,
    flag_sets = [
        flag_set(
            actions = all_compile_actions,
            flag_groups = [
                # For compile actions we have a single source and so put it at
                # the end next to the output.
                flag_group(
                    expand_if_available = "source_file",
                    flags = ["%{source_file}"],
                ),
                flag_group(
                    expand_if_available = "output_file",
                    flags = ["-o", "%{output_file}"],
                ),
            ],
        ),
        flag_set(
            actions = all_link_actions,
            flag_groups = [flag_group(
                expand_if_available = "output_execpath",
                flags = ["-o", "%{output_execpath}"],
            )],
        ),
    ],
)

strip_feature = feature(
    name = "strip_flags",
    enabled = True,
    flag_sets = [flag_set(
        actions = [ACTION_NAMES.strip],
        flag_groups = [
            flag_group(
                flags = ["-S"],
            ),
            flag_group(
                flags = ["-p"],
            ),
            flag_group(
                expand_if_available = "output_file",
                flags = ["-o", "%{output_file}"],
            ),
            flag_group(
                iterate_over = "stripopts",
                flags = ["%{stripopts}"],
            ),
            flag_group(
                expand_if_available = "input_file",
                flags = ["%{input_file}"],
            ),
        ],
    )],
)

base_features = [
    dbg_feature,
    fastbuild_feature,
    host_feature,
    no_legacy_features_feature,
    opt_feature,
    strip_feature,
    supports_pic_feature,
    supports_dynamic_linker_feature,
    supports_start_end_lib_feature,
]
