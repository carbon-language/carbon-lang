# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Definitions of debugging related features used in a `cc_toolchain_config`."""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAME_GROUPS")
load(
    "@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
)

# Handle different levels and forms of debug info emission with individual
# features so that they can be ordered and the defaults can override the
# minimal settings if both are enabled.
minimal_debug_info_flags = feature(
    name = "minimal_debug_info_flags",
    implies = ["debug_info_compression_flags"],
    flag_sets = [flag_set(
        actions = ACTION_NAME_GROUPS.all_cc_compile_actions,
        flag_groups = [flag_group(flags = ["-gmlt"])],
    )],
)
debug_info_flags = feature(
    name = "debug_info_flags",
    implies = ["debug_info_compression_flags"],
    flag_sets = [flag_set(
        actions = ACTION_NAME_GROUPS.all_cc_compile_actions,
        flag_groups = [
            flag_group(flags = ["-g"]),
            flag_group(
                expand_if_available = "per_object_debug_info_file",
                flags = ["-gsplit-dwarf"],
            ),
        ],
    )],
)
debug_info_compression_flags = feature(
    name = "debug_info_compression_flags",
    flag_sets = [flag_set(
        actions = ACTION_NAME_GROUPS.all_cc_compile_actions + ACTION_NAME_GROUPS.all_cc_link_actions,
        flag_groups = [flag_group(flags = ["-gz"])],
    )],
)

# Define a set of mutually exclusive debugger flags.
debugger_flags = feature(name = "debugger_flags")
lldb_flags = feature(
    # Use a convenient name for users to select if needed.
    name = "lldb_flags",
    # Default enable LLDB-optimized flags whenever debugging.
    enabled = True,
    requires = [feature_set(features = ["debug_info_flags"])],
    provides = ["debugger_flags"],
    flag_sets = [flag_set(
        actions = ACTION_NAME_GROUPS.all_cc_compile_actions,
        flag_groups = [flag_group(flags = [
            "-glldb",
            "-gpubnames",
            "-gsimple-template-names",
        ])],
    )],
)
gdb_flags = feature(
    # Use a convenient name for users to select if needed.
    name = "gdb_flags",
    requires = [feature_set(features = ["debug_info_flags"])],
    provides = ["debugger_flags"],
    flag_sets = [
        flag_set(
            actions = ACTION_NAME_GROUPS.all_cc_compile_actions,
            flag_groups = [flag_group(flags = [
                "-ggdb",
                "-ggnu-pubnames",
            ])],
        ),
        flag_set(
            actions = ACTION_NAME_GROUPS.all_cc_link_actions,
            flag_groups = [flag_group(flags = ["-Wl,--gdb-index"])],
        ),
    ],
)

# This feature can be enabled in conjunction with any optimizations to
# ensure accurate call stacks and backtraces for profilers or errors.
preserve_call_stacks = feature(
    name = "preserve_call_stacks",
    flag_sets = [flag_set(
        actions = ACTION_NAME_GROUPS.all_cc_compile_actions,
        flag_groups = [flag_group(flags = [
            # Ensure good backtraces by preserving frame pointers and
            # disabling tail call elimination.
            "-fno-omit-frame-pointer",
            "-mno-omit-leaf-frame-pointer",
            "-fno-optimize-sibling-calls",
        ])],
    )],
)

# Enable split debug info whenever debug info is requested.
enable_split_debug_info = feature(
    name = "per_object_debug_info",
    enabled = True,
    # This has to be directly conditioned on requesting debug info at
    # all, otherwise Bazel will look for an extra output file and not
    # find one.
    requires = [feature_set(features = ["debug_info_flags"])],
)

# Enable debug info whenever in the `dbg` build mode. We do this separately from
# the `debug_info_flags` feature itself as other things may want to enable that
# feature as well.
enable_debug_info_in_dbg = feature(
    name = "enable_debug_info_in_dbg",
    enabled = True,
    requires = [feature_set(["dbg"])],
    implies = ["debug_info_flags"],
)

# Note that the order of features is significant in this list and determines the
# relative order of flags from the features listed.
debugging_features = [
    minimal_debug_info_flags,
    debug_info_flags,
    debug_info_compression_flags,
    debugger_flags,
    lldb_flags,
    gdb_flags,
    preserve_call_stacks,
    enable_split_debug_info,
    enable_debug_info_in_dbg,
]
