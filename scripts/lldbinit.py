#!/usr/bin/env python3

"""Initialization for lldb."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

# This script is only meant to be used from LLDB.
import os
import re
from typing import Any

import lldb  # type: ignore

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
        print("""
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
  ID is an integer number, such as `42`, in hex, such as in `inst6000000A`. It
       can come with a `0x` prefix, allowing easier copy-paste from raw printed
       hex values (such as via the `p/x` lldb command).

Example usage:
  # Dumps the `inst_id` local variable, with a `context` local variable.
  dump context inst_id

  # Dumps the instruction with id 42, with a `context()` method for accessing
  # the `Check::Context&`.
  dump context() inst42
""")

    args = command.split()
    if len(args) < 2:
        print_usage()
        return

    context = args[0]

    # The set of "Make" functions in dump.cpp.
    id_types = {
        "class": "SemIR::MakeClassId",
        "constant": "SemIR::MakeConstantId",
        "constraint": "SemIR::MakeNamedConstraintId",
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
        "require_block": "SemIR::MakeRequireImplsBlockId",
        "require": "SemIR::MakeRequireImplsId",
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
    found_id_type = False

    # Look for <type><id> as a single argument.
    if m := re.fullmatch("([a-z_]+)(?:0x)?([0-9A-Fa-f]+)", args[1]):
        if m[1] in id_types:
            if len(args) > 2:
                print_usage()
                return
            make_id_fn = id_types[m[1]]
            id = int(m[2], 16)
            print_dump(context, f"{make_id_fn}({id})")
            found_id_type = True

    # Look for <type> <id> as two arguments. If there's no <id>, the <type>
    # should just be treated as a variable name.
    if args[1] in id_types:
        if len(args) > 3:
            print_usage()
            return
        elif len(args) == 3:
            if m := re.fullmatch("(?:0x)?([0-9A-Fa-f]+)", args[2]):
                make_id_fn = id_types[args[1]]
                id = int(m[1], 16)
                print_dump(context, f"{make_id_fn}({id})")
                found_id_type = True

    if not found_id_type:
        # Use `--` to escape a variable name like `inst22`.
        if args[1] == "--":
            expr = " ".join(args[2:])
        else:
            expr = " ".join(args[1:])
        print_dump(context, expr)


# Returns true if sbtype is a Carbon ID type (i.e. is derived from
# `Carbon::AnyIdBase`).
def is_carbon_id(sbtype: lldb.SBType, internal_dict: Any) -> bool:
    for base in sbtype.get_bases_array():
        if "Carbon::AnyIdBase" in base.GetName():
            return True
        if is_carbon_id(base.type, internal_dict):
            return True
    return False


# Formats a Carbon ID value to roughly match its format in raw SemIR, without
# calling any user code.
def format_carbon_id(
    valobj: lldb.SBValue, internal_dict: Any, options: Any
) -> str:
    # TODO: It would be safer and more efficient to get these by traversing the
    # member graph using the Python API, rather than by evaluating C++
    # expressions. However, that doesn't seem to work in this case
    # (`SBTypeStaticField.GetConstantValue` seems to be broken), and even if it
    # did, it would be fairly verbose and probably more brittle.
    label = valobj.EvaluateExpression("Label.Data")
    label_size = valobj.EvaluateExpression("Label.Length")

    # For some reason LLDB treats ID types as having an empty `Label` field
    # when accessing an ID via a pointer, so we have to be prepared for that.
    if label and label_size and label_size.GetValueAsUnsigned() > 0:
        # Clamp the read size, to limit the impact of memory corruption.
        # 40 chars should be enough for any legitimate ID label.
        read_size = min(label_size.GetValueAsUnsigned(), 40)
        label_data = valobj.process.ReadMemory(
            label.GetValueAsAddress(), read_size, lldb.SBError()
        )
        label_str = label_data.decode("utf-8")
    else:
        label_str = "<unknown id>"

    index_int = valobj.GetChildMemberWithName("index").GetValueAsUnsigned()
    if index_int == 0xFFFFFFFF:
        # We can't handle all the special cases that ID printing does, but we
        # can at least handle the most common one.
        index_str = "<none>"
    else:
        index_str = f"{index_int:X}"

    return f"{label_str}{index_str}"


def __lldb_init_module(debugger: Any, internal_dict: Any) -> None:
    RunCommand("command script add -f lldbinit.cmd_dump dump")
    RunCommand(
        "type summary add --python-function lldbinit.format_carbon_id"
        + " --recognizer-function lldbinit.is_carbon_id"
    )
