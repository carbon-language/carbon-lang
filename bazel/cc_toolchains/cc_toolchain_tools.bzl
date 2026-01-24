# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Macros to produce tool-related parts of a `cc_toolchain_config`.

These macros cover both the `actions_config` array and the `tool_paths` array.

They presume an LLVM and Clang toolchain's tools, but support both a single
installation and installations that split the LLVM tools and Clang tools apart.
"""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load(
    "@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "action_config",
    "tool",
    "tool_path",
)
load(
    ":cc_toolchain_actions.bzl",
    "all_c_compile_actions",
    "all_cpp_compile_actions",
    "all_link_actions",
)

def llvm_tool_paths(llvm_bindir, clang_bindir = None):
    if not clang_bindir:
        clang_bindir = llvm_bindir
    return [
        tool_path(name = "ar", path = llvm_bindir + "/llvm-ar"),
        tool_path(name = "ld", path = clang_bindir + "/ld.lld"),
        tool_path(name = "cpp", path = clang_bindir + "/clang-cpp"),
        tool_path(name = "gcc", path = clang_bindir + "/clang++"),
        tool_path(name = "dwp", path = llvm_bindir + "/llvm-dwp"),
        tool_path(name = "gcov", path = llvm_bindir + "/llvm-cov"),
        tool_path(name = "nm", path = llvm_bindir + "/llvm-nm"),
        tool_path(name = "objcopy", path = llvm_bindir + "/llvm-objcopy"),
        tool_path(name = "objdump", path = llvm_bindir + "/llvm-objdump"),
        tool_path(name = "strip", path = llvm_bindir + "/llvm-strip"),
    ]

def llvm_action_configs(llvm_bindir, clang_bindir = None):
    if not clang_bindir:
        clang_bindir = llvm_bindir
    return [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tool(path = clang_bindir + "/clang")],
        )
        for name in all_c_compile_actions
    ] + [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tool(path = clang_bindir + "/clang++")],
        )
        for name in all_cpp_compile_actions
    ] + [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tool(path = clang_bindir + "/clang++")],
        )
        for name in all_link_actions
    ] + [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tool(path = llvm_bindir + "/llvm-ar")],
        )
        for name in [ACTION_NAMES.cpp_link_static_library]
    ] + [
        action_config(
            action_name = name,
            enabled = True,
            tools = [tool(path = llvm_bindir + "/llvm-strip")],
        )
        for name in [ACTION_NAMES.strip]
    ]
