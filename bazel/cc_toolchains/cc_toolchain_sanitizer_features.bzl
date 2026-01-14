# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Definitions of sanitizer-related `cc_toolchain_config` features."""

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

sanitizer_common_flags = feature(
    name = "sanitizer_common_flags",
    implies = ["minimal_debug_info_flags", "preserve_call_stacks"],
)

# Separated from the feature above so it can only be included on platforms
# where it is supported. There is no negative flag in Clang so we can't just
# override it later.
sanitizer_static_lib_flags = feature(
    name = "sanitizer_static_lib_flags",
    enabled = True,
    requires = [feature_set(["sanitizer_common_flags"])],
    flag_sets = [flag_set(
        actions = all_link_actions,
        flag_groups = [flag_group(flags = ["-static-libsan"])],
    )],
)

asan = feature(
    name = "asan",
    implies = ["sanitizer_common_flags"],
    flag_sets = [flag_set(
        actions = all_compile_actions + all_link_actions,
        flag_groups = [flag_group(flags = [
            "-fsanitize=address,undefined,nullability",
            "-fsanitize-address-use-after-scope",
            # Outlining is almost always the right tradeoff for our
            # sanitizer usage where we're more pressured on generated code
            # size than runtime performance.
            "-fsanitize-address-outline-instrumentation",
            # We don't need the recovery behavior of UBSan as we expect
            # builds to be clean. Not recovering is a bit cheaper.
            "-fno-sanitize-recover=undefined,nullability",
            # Don't embed the full path name for files. This limits the size
            # and combined with line numbers is unlikely to result in many
            # ambiguities.
            "-fsanitize-undefined-strip-path-components=-1",
            # Needed due to clang AST issues, such as in
            # clang/AST/Redeclarable.h line 199.
            "-fno-sanitize=vptr",
        ])],
    )],
)

# A feature that further reduces the generated code size of our the ASan
# feature, but at the cost of lower quality diagnostics. This is enabled
# along with ASan in our fastbuild configuration, but can be disabled
# explicitly to get better error messages.
asan_min_size = feature(
    name = "asan_min_size",
    requires = [feature_set(["asan"])],
    flag_sets = [flag_set(
        actions = all_compile_actions + all_link_actions,
        flag_groups = [flag_group(flags = [
            # Force two UBSan checks that have especially large code size
            # cost to use the minimal branch to a trapping instruction model
            # instead of the full diagnostic.
            "-fsanitize-trap=alignment,null",
        ])],
    )],
)

# Likely due to being unable to use the static-linked and up-to-date
# sanitizer runtimes, we have to disable a number of sanitizers on macOS.
macos_asan_workarounds = feature(
    name = "macos_sanitizer_workarounds",
    enabled = True,
    requires = [feature_set(["asan"])],
    flag_sets = [flag_set(
        actions = all_compile_actions + all_link_actions,
        flag_groups = [flag_group(flags = [
            "-fno-sanitize=function",
        ])],
    )],
)

fuzzer = feature(
    name = "fuzzer",
    flag_sets = [flag_set(
        actions = all_compile_actions + all_link_actions,
        flag_groups = [flag_group(flags = [
            "-fsanitize=fuzzer-no-link",
        ])],
    )],
)

# Note that the order of features is significant in this list and determines the
# relative order of flags from the features listed.
sanitizer_features = [
    sanitizer_common_flags,
    asan,
    asan_min_size,
    fuzzer,
]
