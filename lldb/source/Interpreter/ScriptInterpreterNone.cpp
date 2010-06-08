//===-- ScriptInterpreterNone.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/ScriptInterpreterNone.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StringList.h"

using namespace lldb;
using namespace lldb_private;

ScriptInterpreterNone::ScriptInterpreterNone () :
    ScriptInterpreter (eScriptLanguageNone)
{
}

ScriptInterpreterNone::~ScriptInterpreterNone ()
{
}

void
ScriptInterpreterNone::ExecuteOneLine (const std::string &line, FILE *out, FILE *err)
{
    ::fprintf (err, "error: there is no embedded script interpreter in this mode.\n");
}

void
ScriptInterpreterNone::ExecuteInterpreterLoop (FILE *out, FILE *err)
{
    fprintf (err, "error: there is no embedded script interpreter in this mode.\n");
}


