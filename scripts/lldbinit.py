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
        print(
            """
Dumps the value of an associated ID, using the C++ Dump() functions.

Usage:
  dump <CONTEXT> [<EXPR>|-- <EXPR>|<TYPE><ID>|<TYPE> <ID>]

Args:
  CONTEXT is the dump context, such a SemIR::Context reference, a SemIR::File,
          a Parse::Context, or a Lex::TokenizeBuffer.
  EXPR is a C++ expression such as a variable name. Use `--` to prevent it from
       being treated as a TYPE and ID.
  TYPE can be `inst`, `constant`, `generic`, `impl`, `entity_name`, etc. See
       the `Label` string in `IdBase` classes to find possible TYPE names,
       though only Id types that have a matching `Make...Id()` function are
       supported.
  ID is an integer number, such as `42`.

Example usage:
  # Dumps the `inst_id` local variable, with a `context` local variable.
  dump context inst_id

  # Dumps the instruction with id 42, with a `context()` method for accessing
  # the `Check::Context&`.
  dump context() inst42
"""
        )

    args = command.split(" ")
    if len(args) < 2:
        print_usage()
        return

    context = args[0]

    # The set of "Make" functions in dump.cpp.
    id_types = {
        "class": "SemIR::MakeClassId",
        "constant": "SemIR::MakeConstantId",
        "symbolic_constant": "SemIR::MakeSymbolicConstantId",
        "entity_name": "SemIR::MakeEntityNameId",
        "facet_type": "SemIR::MakeFacetTypeId",
        "function": "SemIR::MakeFunctionId",
        "generic": "SemIR::MakeGenericId",
        "impl": "SemIR::MakeImplId",
        "inst_block": "SemIR::MakeInstBlockId",
        "inst": "SemIR::MakeInstId",
        "interface": "SemIR::MakeInterfaceId",
        "name": "SemIR::MakeNameId",
        "name_scope": "SemIR::MakeNameScopeId",
        "identified_facet_type": "SemIR::MakeIdentifiedFacetTypeId",
        "specific": "SemIR::MakeSpecificId",
        "specific_interface": "SemIR::MakeSpecificInterfaceId",
        "struct_type_fields": "SemIR::MakeStructTypeFieldsId",
        "type": "SemIR::MakeTypeId",
    }

    def print_dump(context: str, expr: str) -> None:
        cmd = f"p Dump({context}, {expr})"
        out = RunCommand(cmd, print_command=False)
        if m := re.match(dump_re, out):
            # Use the `dump_re` match to print just the interesting part of the
            # dump output.
            print(m[1])
        else:
            # Unexpected output, show the command that was run.
            print(f"(lldb) {cmd}")
            print(out)

    # Try to find a type + id from the input args. If not, the id will be passed
    # through directly to C++, as it can be a variable name.
    id_type = None

    # Look for <type><id> as a single argument.
    if m := re.fullmatch("([a-z_]+)(\\d+)", args[1]):
        if m[1] in id_types:
            if len(args) != 2:
                print_usage()
                return
            id_type = m[1]
            print_dump(context, f"{id_types[id_type]}({m[2]})")

    # Look for <type> <id> as two arguments.
    if not id_type:
        if args[1] in id_types:
            if len(args) != 3:
                print_usage()
                return
            id_type = args[1]
            print_dump(context, f"{id_types[id_type]}({args[2]})")

    if not id_type:
        # Use `--` to escape a variable name like `inst22`.
        if args[1] == "--":
            expr = " ".join(args[2:])
        else:
            expr = " ".join(args[1:])
        print_dump(context, expr)


def __lldb_init_module(debugger: Any, internal_dict: Any) -> None:
    RunCommand("command script add -f lldbinit.cmd_dump dump")
