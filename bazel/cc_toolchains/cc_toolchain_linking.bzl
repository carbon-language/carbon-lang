# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Definitions of linking related features used in a `cc_toolchain_config`."""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load(
    "@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "variable_with_value",
    "with_feature_set",
)
load(
    ":cc_toolchain_actions.bzl",
    "all_link_actions",
)

link_libraries_feature = feature(
    name = "link_libraries",
    enabled = True,
    flag_sets = [
        flag_set(
            actions = all_link_actions,
            flag_groups = [
                flag_group(
                    expand_if_available = "linkstamp_paths",
                    flags = ["%{linkstamp_paths}"],
                    iterate_over = "linkstamp_paths",
                ),
                flag_group(
                    expand_if_available = "libraries_to_link",
                    flag_groups = [
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "object_file_group",
                            ),
                            flags = ["-Wl,--start-lib"],
                        ),
                        flag_group(
                            expand_if_true = "libraries_to_link.is_whole_archive",
                            flags = ["-Wl,-whole-archive"],
                        ),
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "object_file_group",
                            ),
                            flags = ["%{libraries_to_link.object_files}"],
                            iterate_over = "libraries_to_link.object_files",
                        ),
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "object_file",
                            ),
                            flags = ["%{libraries_to_link.name}"],
                        ),
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "interface_library",
                            ),
                            flags = ["%{libraries_to_link.name}"],
                        ),
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "static_library",
                            ),
                            flags = ["%{libraries_to_link.name}"],
                        ),
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "dynamic_library",
                            ),
                            flags = ["-l%{libraries_to_link.name}"],
                        ),
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "versioned_dynamic_library",
                            ),
                            flags = ["-l:%{libraries_to_link.name}"],
                        ),
                        flag_group(
                            expand_if_true = "libraries_to_link.is_whole_archive",
                            flags = ["-Wl,-no-whole-archive"],
                        ),
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "object_file_group",
                            ),
                            flags = ["-Wl,--end-lib"],
                        ),
                    ],
                    iterate_over = "libraries_to_link",
                ),
                # Note that the params file comes at the end, after the
                # libraries to link above.
                flag_group(
                    expand_if_available = "linker_param_file",
                    flags = ["@%{linker_param_file}"],
                ),
            ],
            with_features = [with_feature_set(not_features = ["macos_target"])],
        ),
        flag_set(
            actions = all_link_actions,
            flag_groups = [
                flag_group(
                    expand_if_available = "linkstamp_paths",
                    flags = ["%{linkstamp_paths}"],
                    iterate_over = "linkstamp_paths",
                ),
                flag_group(
                    expand_if_available = "libraries_to_link",
                    flag_groups = [
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "object_file_group",
                            ),
                            flags = ["-Wl,--start-lib"],
                        ),
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "object_file_group",
                            ),
                            flag_groups = [
                                flag_group(
                                    expand_if_false = "libraries_to_link.is_whole_archive",
                                    flags = ["%{libraries_to_link.object_files}"],
                                ),
                                flag_group(
                                    expand_if_true = "libraries_to_link.is_whole_archive",
                                    flags = ["-Wl,-force_load,%{libraries_to_link.object_files}"],
                                ),
                            ],
                            iterate_over = "libraries_to_link.object_files",
                        ),
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "object_file",
                            ),
                            flag_groups = [
                                flag_group(
                                    expand_if_false = "libraries_to_link.is_whole_archive",
                                    flags = ["%{libraries_to_link.name}"],
                                ),
                                flag_group(
                                    expand_if_true = "libraries_to_link.is_whole_archive",
                                    flags = ["-Wl,-force_load,%{libraries_to_link.name}"],
                                ),
                            ],
                        ),
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "interface_library",
                            ),
                            flag_groups = [
                                flag_group(
                                    expand_if_false = "libraries_to_link.is_whole_archive",
                                    flags = ["%{libraries_to_link.name}"],
                                ),
                                flag_group(
                                    expand_if_true = "libraries_to_link.is_whole_archive",
                                    flags = ["-Wl,-force_load,%{libraries_to_link.name}"],
                                ),
                            ],
                        ),
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "static_library",
                            ),
                            flag_groups = [
                                flag_group(
                                    expand_if_false = "libraries_to_link.is_whole_archive",
                                    flags = ["%{libraries_to_link.name}"],
                                ),
                                flag_group(
                                    expand_if_true = "libraries_to_link.is_whole_archive",
                                    flags = ["-Wl,-force_load,%{libraries_to_link.name}"],
                                ),
                            ],
                        ),
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "dynamic_library",
                            ),
                            flags = ["-l%{libraries_to_link.name}"],
                        ),
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "versioned_dynamic_library",
                            ),
                            flags = ["-l:%{libraries_to_link.name}"],
                        ),
                        flag_group(
                            expand_if_true = "libraries_to_link.is_whole_archive",
                            flag_groups = [
                                flag_group(
                                    expand_if_false = "macos_flags",
                                    flags = ["-Wl,-no-whole-archive"],
                                ),
                            ],
                        ),
                        flag_group(
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "object_file_group",
                            ),
                            flags = ["-Wl,--end-lib"],
                        ),
                    ],
                    iterate_over = "libraries_to_link",
                ),
                # Note that the params file comes at the end, after the
                # libraries to link above.
                flag_group(
                    expand_if_available = "linker_param_file",
                    flags = ["@%{linker_param_file}"],
                ),
            ],
            with_features = [with_feature_set(["macos_target"])],
        ),
    ],
)

# Archive actions have an entirely independent set of flags and don't
# interact with either compiler or link actions.
archiving_feature = feature(
    name = "archiving",
    enabled = True,
    flag_sets = [flag_set(
        actions = [ACTION_NAMES.cpp_link_static_library],
        flag_groups = [
            flag_group(flags = ["rcsD"]),
            flag_group(
                expand_if_available = "output_execpath",
                flags = ["%{output_execpath}"],
            ),
            flag_group(
                expand_if_available = "libraries_to_link",
                flag_groups = [
                    flag_group(
                        expand_if_equal = variable_with_value(
                            name = "libraries_to_link.type",
                            value = "object_file",
                        ),
                        flags = ["%{libraries_to_link.name}"],
                    ),
                    flag_group(
                        expand_if_equal = variable_with_value(
                            name = "libraries_to_link.type",
                            value = "object_file_group",
                        ),
                        flags = ["%{libraries_to_link.object_files}"],
                        iterate_over = "libraries_to_link.object_files",
                    ),
                ],
                iterate_over = "libraries_to_link",
            ),
            flag_group(
                expand_if_available = "linker_param_file",
                flags = ["@%{linker_param_file}"],
            ),
        ],
    )],
)

# Note that the order of features is significant in this list and determines the
# relative order of flags from the features listed.
linking_features = [
    link_libraries_feature,
    archiving_feature,
]
