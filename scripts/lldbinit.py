#!/usr/bin/env python3

"""Initialization for lldb."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

# This script is only meant to be used from LLDB.
import lldb  # type: ignore
import os

project_root = os.path.dirname(os.path.realpath(__file__))

ci = lldb.debugger.GetCommandInterpreter()
result = lldb.SBCommandReturnObject()


def RunCommand(cmd: str) -> None:
    """Runs a command and prints it to the console to show that it ran."""
    print("(lldb) %s" % cmd)
    ci.HandleCommand(cmd, result)


RunCommand(f"settings append target.source-map . {project_root}")
RunCommand(f"settings append target.source-map /proc/self/cwd {project_root}")
