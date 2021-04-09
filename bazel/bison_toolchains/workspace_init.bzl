# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Workspace repositories supporting Bison and Flex."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def bison_workspace_init():
    # TODO(chandlerc): Replace this with an upstream release once the pull request
    # with our needed functionality lands:
    # https://github.com/jmillikin/rules_m4/pull/7
    #
    # Until then, this is pulling from that pull request's commit.
    http_archive(
        name = "rules_m4",
        sha256 = "4d34917214e8890ad770bdf0c319c41c9201fffd770938b41a1d641d4b27e05c",
        strip_prefix = "rules_m4-add-extra-copts",
        urls = ["https://github.com/chandlerc/rules_m4/archive/add-extra-copts.zip"],
    )

    # TODO(chandlerc): Replace this with an upstream release once the pull request
    # with our needed functionality lands:
    # https://github.com/jmillikin/rules_flex/pull/5
    #
    # Until then, this is pulling from that pull request's commit.
    http_archive(
        name = "rules_flex",
        sha256 = "fd97c3ae23926507be1b95158a683cd41c628d201e852a325d38b5e9f821b752",
        strip_prefix = "rules_flex-add-extra-copts",
        urls = ["https://github.com/chandlerc/rules_flex/archive/add-extra-copts.zip"],
    )

    # TODO(chandlerc): Replace this with an upstream release once the pull request
    # with our needed functionality lands:
    # https://github.com/jmillikin/rules_bison/pull/7
    #
    # Until then, this is pulling from that pull request's commit.
    http_archive(
        name = "rules_bison",
        sha256 = "c6e926f15214d903966dc950d759ec69116db67f148be114c119e4def0551eaa",
        strip_prefix = "rules_bison-add-extra-copts",
        urls = ["https://github.com/chandlerc/rules_bison/archive/add-extra-copts.zip"],
    )
