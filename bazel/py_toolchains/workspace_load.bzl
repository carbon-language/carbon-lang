# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Workspace rules supporting Python."""

load("@rules_python//python:pip.bzl", "pip_install")

def py_workspace_load():
    # Create a central repo that knows about the pip dependencies.
    pip_install(
        name = "py_deps",
        requirements = "//github_tools:requirements.txt",
    )
