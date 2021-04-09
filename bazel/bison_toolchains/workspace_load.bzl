# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Workspace rules supporting Bison and Flex."""

load("@rules_bison//bison:bison.bzl", "bison_register_toolchains")
load("@rules_flex//flex:flex.bzl", "flex_register_toolchains")
load("@rules_m4//m4:m4.bzl", "m4_register_toolchains")

def bison_workspace_load():
    # When building M4, disable all compiler warnings as we can't realistically fix
    # them anyways.
    m4_register_toolchains(extra_copts = ["-w"])

    # When building Flex, disable all compiler warnings as we can't realistically
    # fix them anyways.
    flex_register_toolchains(extra_copts = ["-w"])

    # When building Bison, disable all compiler warnings as we can't realistically
    # fix them anyways.
    bison_register_toolchains(extra_copts = ["-w"])
