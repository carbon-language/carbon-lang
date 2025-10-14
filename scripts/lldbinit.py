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
dump_re = re.compile(r'\(std::string\) "([\s\S]+)"', re.MULTILINE)


# A helper to ease calling the Dump() free functions.
def cmd_dump(debugger: Any, command: Any, result: Any, dict: Any) -> None:
    def print_usage() -> None:
        print(
            """
Dumps the value of an associated ID, using the C++ Dump() functions.

Usage:
  dump <CONTEXT> [<EXPR>|-- <EXPR>|<TYPE><ID>]

Args:
  CONTEXT is the dump context, such a SemIR::Context reference, a SemIR::File,
          a Parse::Context, or a Lex::TokenizeBuffer.
  EXPR is a C++ expression such as a variable name. Use `--` to prevent it from
       being treated as a TYPE and ID.
  TYPE can be `inst`, `constant`, `generic`, `impl`, `entity_name`, etc. See
       the `Label` string in `IdBase` classes to find possible TYPE names,
       though only Id types that have a matching `Make...Id()` function are
       supported.
  ID is an integer number, such as `42`. This can be in hex (without the typical
       0x prefix), such as `6000A`.

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

    DECIMAL = 10
    HEX = 16

    # The set of "Make" functions in dump.cpp, and whether the ids are printed
    # in decimal or hex.
    id_types = {
        "class": ("SemIR::MakeClassId", HEX),
        "constant": ("SemIR::MakeConstantId", DECIMAL),
        "symbolic_constant": ("SemIR::MakeSymbolicConstantId", DECIMAL),
        "entity_name": ("SemIR::MakeEntityNameId", DECIMAL),
        "facet_type": ("SemIR::MakeFacetTypeId", DECIMAL),
        "function": ("SemIR::MakeFunctionId", HEX),
        "generic": ("SemIR::MakeGenericId", DECIMAL),
        "impl": ("SemIR::MakeImplId", HEX),
        "inst_block": ("SemIR::MakeInstBlockId", DECIMAL),
        "inst": ("SemIR::MakeInstId", HEX),
        "interface": ("SemIR::MakeInterfaceId", DECIMAL),
        "name": ("SemIR::MakeNameId", DECIMAL),
        "name_scope": ("SemIR::MakeNameScopeId", DECIMAL),
        "identified_facet_type": ("SemIR::MakeIdentifiedFacetTypeId", DECIMAL),
        "specific": ("SemIR::MakeSpecificId", DECIMAL),
        "specific_interface": ("SemIR::MakeSpecificInterfaceId", HEX),
        "struct_type_fields": ("SemIR::MakeStructTypeFieldsId", DECIMAL),
        "type": ("SemIR::MakeTypeId", DECIMAL),
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
    found_id_type = False

    # Look for <type><id> as a single argument.
    if m := re.fullmatch("([a-z_]+)([0-9A-Fa-f]+)", args[1]):
        if m[1] in id_types:
            if len(args) != 2:
                print_usage()
                return
            (make_id_fn, base) = id_types[m[1]]
            id = int(m[2], base)
            print_dump(context, f"{make_id_fn}({id})")
            found_id_type = True

    if not found_id_type:
        # Use `--` to escape a variable name like `inst22`.
        if args[1] == "--":
            expr = " ".join(args[2:])
        else:
            expr = " ".join(args[1:])
        print_dump(context, expr)


def __lldb_init_module(debugger: Any, internal_dict: Any) -> None:
    RunCommand("command script add -f lldbinit.cmd_dump dump")
