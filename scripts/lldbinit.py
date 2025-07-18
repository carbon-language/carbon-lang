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
import re
from typing import Any

project_root = os.path.dirname(os.path.realpath(__file__))

ci = lldb.debugger.GetCommandInterpreter()
result = lldb.SBCommandReturnObject()


def RunCommand(cmd: str, print_command: bool = True) -> Any:
    """Runs a command and prints it to the console to show that it ran."""
    if print_command:
        print(f"(lldb) {cmd}")
    ci.HandleCommand(cmd, result)
    return result.GetOutput()


RunCommand(f"settings append target.source-map . {project_root}")
RunCommand(f"settings append target.source-map /proc/self/cwd {project_root}")

# Matches the output of `print Dump(...)` and captures the stuff from inside the
# std::string while discarding the std::string type.
dump_re = re.compile('\\(std::string\\) "((:?.|\n)+)"', re.MULTILINE)


# A helper to ease calling the Dump() free functions.
def cmd_dump(debugger: Any, command: Any, result: Any, dict: Any) -> None:
    def print_usage() -> None:
        print("dump <context> [<ID>|<TYPE><ID>|<TYPE> <ID>|-- <ID>]")
        print("")
        print('TYPE can be "inst", "entity_name", etc.')

    args = command.split(" ")
    if len(args) < 2:
        print_usage()
        return

    context = args[0]

    # The set of "Make" functions in dump.cpp.
    id_types = [
        ("class", "MakeClassId"),
        ("constant", "MakeConstantId"),
        ("symbolic_constant", "MakeSymbolicConstantId"),
        ("entity_name", "MakeEntityNameId"),
        ("facet_type", "MakeFacetTypeId"),
        ("function", "MakeFunctionId"),
        ("generic", "MakeGenericId"),
        ("impl", "MakeImplId"),
        ("inst_block", "MakeInstBlockId"),
        ("inst", "MakeInstId"),
        ("interface", "MakeInterfaceId"),
        ("name", "MakeNameId"),
        ("name_scope", "MakeNameScopeId"),
        ("identified_facet_type", "MakeIdentifiedFacetTypeId"),
        ("specific", "MakeSpecificId"),
        ("specific_interface", "MakeSpecificInterfaceId"),
        ("struct_type_fields", "MakeStructTypeFieldsId"),
        ("type", "MakeTypeId"),
    ]

    # Try find a type + id from the input args. If not, the id will be passed
    # through directly to C++, as it can be a variable name.
    id_type = None

    # Look for <type><id> as a single argument.
    for id_type_pair in id_types:
        if re.fullmatch(f"{id_type_pair[0]}[0-9]+", args[1]):
            at = len(id_type_pair[0])
            id_type = args[1][:at]
            id = args[1][at:]
            break

    # Look for <type> <id> as two arguments.
    if not id_type:
        for id_type_pair in id_types:
            if id_type_pair[0] == args[1]:
                if len(args) < 3:
                    print_usage()
                    return
                id_type = args[1]
                id = args[2]
                break

    # If we have an id type, transform the id as a number into that id
    # with the factory function.
    if id_type:
        for id_type_pair in id_types:
            if id_type_pair[0] == id_type:
                id = f"SemIR::{id_type_pair[1]}({id})"
                break
    else:
        # Use `--` to escape a variable name like `inst22`.
        if args[1] == "--":
            id = " ".join(args[2:])
        else:
            id = " ".join(args[1:])

    cmd = f"p Dump({context}, {id})"
    out = RunCommand(cmd, print_command=False)
    m = re.match(dump_re, out)
    if m:
        print(m.group(1))
    else:
        # Unexpected output, show the command that was run.
        print(f"(lldb) {cmd}")
        print(out)


def __lldb_init_module(debugger: Any, internal_dict: Any) -> None:
    RunCommand("command script add -f lldbinit.cmd_dump dump")
