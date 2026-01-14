# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Definitions of C++ and Clang header modules toolchain features."""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load(
    "@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
)

use_module_maps = feature(
    name = "use_module_maps",
    requires = [feature_set(features = ["module_maps"])],
    flag_sets = [
        flag_set(
            actions = [
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.cpp_header_parsing,
                ACTION_NAMES.cpp_module_compile,
            ],
            flag_groups = [
                # These flag groups are separate so they do not expand to
                # the cross product of the variables.
                flag_group(flags = ["-fmodule-name=%{module_name}"]),
                flag_group(
                    flags = ["-fmodule-map-file=%{module_map_file}"],
                ),
            ],
        ),
    ],
)

# Tell bazel we support module maps in general, so they will be generated
# for all c/c++ rules.
# Note: not all C++ rules support module maps; thus, do not imply this
# feature from other features - instead, require it.
module_maps = feature(
    name = "module_maps",
    enabled = True,
    implies = [
        # "module_map_home_cwd",
        # "module_map_without_extern_module",
        # "generate_submodules",
    ],
)

layering_check = feature(
    name = "layering_check",
    implies = ["use_module_maps"],
    flag_sets = [flag_set(
        actions = [
            ACTION_NAMES.c_compile,
            ACTION_NAMES.cpp_compile,
            ACTION_NAMES.cpp_header_parsing,
            ACTION_NAMES.cpp_module_compile,
        ],
        flag_groups = [
            flag_group(flags = [
                "-fmodules-strict-decluse",
                "-Wprivate-header",
            ]),
            flag_group(
                iterate_over = "dependent_module_map_files",
                flags = ["-fmodule-map-file=%{dependent_module_map_files}"],
            ),
        ],
    )],
)

# Note that the order of features is significant in this list and determines the
# relative order of flags from the features listed.
modules_features = [
    layering_check,
    module_maps,
    use_module_maps,
]
