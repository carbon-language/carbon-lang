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

os_names = [
    "freebsd",
    "linux",
    "macos",
    "windows",
]

def target_os_features(target_os):
    if target_os not in os_names:
        fail("Unsupported target OS: %s" % target_os)
    return [
        feature(name = os_name + "_target", enabled = os_name == target_os)
        for os_name in os_names
    ]

cpu_names = [
    "aarch64",
    "x86_64",
]

# Also support canonicalizing different spellings of CPUs to one of the above
# names.
cpu_canonical_name_map = {
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "x86_64": "x86_64",
}

def target_cpu_features(target_cpu):
    if target_cpu not in cpu_canonical_name_map:
        fail("Unsupported target CPU: %s" % target_cpu)
    target_cpu = cpu_canonical_name_map[target_cpu]

    return [
        feature(name = cpu_name + "_target", enabled = cpu_name == target_cpu)
        for cpu_name in cpu_names
    ]
