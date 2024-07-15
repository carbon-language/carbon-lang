# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compute the version string."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("//:version_base.bzl", "version_base")

def _validate_nightly_date(date):
    date_components = date.split(".", 2)
    if len(date_components) != 3:
        fail("Must provide a nightly date in 'YYYY.MM.DD' format, found '{}'.".format(date))
    year = date_components[0]
    if len(year) != 4 or not year.isdigit():
        fail("The nightly date year was not a sequence of four digits.")
    month = date_components[1]
    if len(month) != 2 or not month.isdigit():
        fail("The nightly date month was not a sequence of two digits.")
    day = date_components[2]
    if len(day) != 2 or not day.isdigit():
        fail("The nightly date day was not a sequence of two digits.")

def compute_version(ctx):
    """Compute the version string.

    Args:
        ctx: The context for a rule computing the version.

    Returns:
        The version string.
    """
    version = version_base

    # See if we need to append a pre-release suffix to the version.
    #
    # TODO: We should more fully check for erroneous combinations of flags here
    # to help ensure users don't get surprising results.
    if not ctx.attr._release_flag[BuildSettingInfo].value:
        pre_release = ctx.attr._pre_release_flag[BuildSettingInfo].value
        pre_release_numbers = {
            "rc": ctx.attr._rc_number_flag[BuildSettingInfo].value,
        }
        if pre_release in pre_release_numbers:
            number = pre_release_numbers[pre_release]
            if number < 0:
                fail("Must provide a non-negative {} number when building that pre-release.".format(pre_release))
            version += "-{0}.{1}".format(pre_release, number)
        elif pre_release == "nightly":
            date = ctx.attr._nightly_date_flag[BuildSettingInfo].value
            _validate_nightly_date(date)
            version += "-0.nightly.{}".format(date)
        elif pre_release == "dev":
            version += "-0.dev"
        else:
            fail("Invalid pre-release flag: " + pre_release)

    return version

VERSION_ATTRS = {
    "_nightly_date_flag": attr.label(default = ":nightly_date"),
    "_pre_release_flag": attr.label(default = ":pre_release"),
    "_rc_number_flag": attr.label(default = ":rc_number"),
    "_release_flag": attr.label(default = ":release"),
}
