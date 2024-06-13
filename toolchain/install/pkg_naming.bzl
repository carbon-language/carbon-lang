# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule to create variables for package naming."""

load("@rules_pkg//pkg:providers.bzl", "PackageVariablesInfo")
load("//bazel/version:compute_version.bzl", "VERSION_ATTRS", "compute_version")

def _pkg_naming_variables_impl(ctx):
    # TODO: Add support for digging the target CPU out of the toolchain here,
    # remapping it to a more canonical name, and add that to the variables. The
    # Bazel target CPU is already directly available, but it isn't likely
    # canonical.
    # TODO: Include the target OS as well as the target CPU. This likely needs
    # similar re-mapping as the CPU does.
    return PackageVariablesInfo(values = {
        "version": compute_version(ctx),
    })

pkg_naming_variables = rule(
    implementation = _pkg_naming_variables_impl,
    attrs = VERSION_ATTRS,
)
