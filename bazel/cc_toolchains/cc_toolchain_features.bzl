# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helpers to construct ordered sequences of `cc_toolchain` features."""

load(
    ":cc_toolchain_base_features.bzl",
    "base_features",
    "output_flags_feature",
    "user_flags_feature",
)
load(
    ":cc_toolchain_config_features.bzl",
    "target_cpu_features",
    "target_os_features",
)
load(
    ":cc_toolchain_cpp_features.bzl",
    "clang_feature",
    "clang_warnings_feature",
)
load(":cc_toolchain_debugging.bzl", "debugging_features")
load(":cc_toolchain_linking.bzl", "linking_features")
load(":cc_toolchain_modules.bzl", "modules_features")
load(":cc_toolchain_optimization.bzl", "optimization_features")
load(":cc_toolchain_sanitizer_features.bzl", "sanitizer_features")

def clang_cc_toolchain_features(
        target_os,
        target_cpu,
        project_features = [],
        extra_cpp_features = []):
    """Builds a sequence of Clang-oriented `cc_toolchain_config` features.

    Returns:
        The list of features for calling `create_cc_toolchain_config_info`.

    Args:
        target_os: Used to select OS-specific features to include.
        target_cpu: Used to select CPU-specific features to include.
        project_features: Optional list of project-specific features to include.
        extra_cpp_features: Optional list of extra C++ features to include, for
            example `libcxx_feature` can be passed here to enable using libc++.
    """

    # The order of the features determines the relative order of flags used.
    features = []
    features += target_os_features(target_os)
    features += target_cpu_features(target_cpu)
    features += base_features
    features += [
        # We always use Clang in the toolchain and enable all of its warnings.
        clang_feature,
        clang_warnings_feature,
    ]

    # Enable any extra baseline C++ features here where others can override
    # their flags if needed.
    features += extra_cpp_features

    features += sanitizer_features
    features += optimization_features
    features += modules_features
    features += debugging_features
    features += linking_features

    # Lastly, we add project features and the user flags so they can override
    # anything above, and the output flags last of all for ease of debugging.
    features += project_features
    features += [
        user_flags_feature,
        output_flags_feature,
    ]
    return features
