# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

workspace(name = "carbon")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Adds @rules_python.
load("//bazel/py_toolchains:workspace_init.bzl", "py_workspace_init")

py_workspace_init()

# Loads Python packages.
load("//bazel/py_toolchains:workspace_load.bzl", "py_workspace_load")

py_workspace_load()

# Adds @llvm_bazel for C++.
local_repository(
    name = "llvm_bazel",
    path = "third_party/llvm-bazel/llvm-bazel",
)

# Loads C++ toolchains and @llvm-project.
load("//bazel/cc_toolchains:workspace_load.bzl", "cc_workspace_load")

cc_workspace_load()

# Adds @rules_bison, @rules_flex, and @rules_m4.
load("//bazel/bison_toolchains:workspace_init.bzl", "bison_workspace_init")

bison_workspace_init()

# Loads Bison and Flex toolchains.
load("//bazel/bison_toolchains:workspace_load.bzl", "bison_workspace_load")

bison_workspace_load()
