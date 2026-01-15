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

freebsd_target_feature = feature(name = "freebsd_target", enabled = True)
linux_target_feature = feature(name = "linux_target", enabled = True)
macos_target_feature = feature(name = "macos_target", enabled = True)
windows_target_feature = feature(name = "windows_target", enabled = True)

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

aarch64_target_feature = feature(name = "aarch64_target", enabled = True)
x86_64_target_feature = feature(name = "x86_64_target", enabled = True)

cpu_target_features = {
    "aarch64": [aarch64_target_feature],
    "arm64": [aarch64_target_feature],
    "x86_64": [x86_64_target_feature],
}

def target_cpu_features(cpu):
    if cpu not in cpu_target_features:
        fail("Unsupported target CPU: %s" % cpu)

    return cpu_target_features[cpu]
