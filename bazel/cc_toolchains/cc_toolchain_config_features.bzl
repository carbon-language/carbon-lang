# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configuration features for other features in a `cc_toolchain_config`.

These features are designed to be used by other features in a
`cc_toolchain_config` that need to configure their behavior in some way. This
can be configuration based on either the target or host of the build, and along
multiple dimensions of each.
"""

load(
    "@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "feature",
)

freebsd_target_feature = feature(name = "freebsd_target")
linux_target_feature = feature(name = "linux_target")
macos_target_feature = feature(name = "macos_target")
windows_target_feature = feature(name = "windows_target")

os_target_features = {
    "freebsd": [freebsd_target_feature],
    "linux": [linux_target_feature],
    "macos": [macos_target_feature],
    "windows": [windows_target_feature],
}

def target_os_features(os):
    if os not in os_target_features:
        fail("Unsupported target OS: %s" % os)

    return os_target_features[os]
