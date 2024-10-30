# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule to create variables for package naming."""

load("@rules_pkg//pkg:providers.bzl", "PackageVariablesInfo")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_test")
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

def pkg_tar_and_test(name_base, package_file_name_base, install_data_manifest, **kwargs):
    """Create a `pkg_tar` and a test for both `.tar` and `.tar.gz` extensions.

    Args:
        name_base:
            The base name of the rules and tests. Will have `tar` or `tar_gz` added
            and then `_rule` for the `pkg_tar` and `_test` for the test.
        package_file_name_base:
            The base of the `package_file_name` attribute to `pkg_tar`. The file
            extensions will be appended after a `.`.
        install_data_manifest:
            The install data manifest file to compare with.
        **kwargs:
            Passed to `pkg_tar` for all the rest of its attributes.
    """
    for file_ext in ["tar", "tar.gz"]:
        target_ext = file_ext.replace(".", "_")
        tar_target = name_base + "_" + target_ext + "_rule"
        pkg_tar(
            name = tar_target,
            extension = file_ext,
            package_file_name = package_file_name_base + "." + file_ext,
            # The compressed tar is slow, exclude building and testing that.
            tags = ["manual"] if file_ext == "tar.gz" else [],
            **kwargs
        )

        py_test(
            name = name_base + "_" + target_ext + "_test",
            size = "small",
            srcs = ["toolchain_tar_test.py"],
            data = [":" + tar_target, install_data_manifest],
            env = {
                "INSTALL_DATA_MANIFEST": "$(location {})".format(install_data_manifest),
                "TAR_FILE": "$(location :{})".format(tar_target),
            },
            main = "toolchain_tar_test.py",
            # The compressed tar is slow, exclude building and testing that.
            tags = ["manual"] if file_ext == "tar.gz" else [],
        )
