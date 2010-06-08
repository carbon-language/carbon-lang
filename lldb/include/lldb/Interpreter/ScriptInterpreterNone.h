//===-- ScriptInterpreterNone.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ScriptInterpreterNone_h_
#define liblldb_ScriptInterpreterNone_h_

#include "lldb/Interpreter/ScriptInterpreter.h"

namespace lldb_private {

class ScriptInterpreterNone : public ScriptInterpreter
{
public:

    ScriptInterpreterNone ();

    ~ScriptInterpreterNone ();

    virtual void
    ExecuteOneLine (const std::string &line, FILE *out, FILE *err);

    virtual void
    ExecuteInterpreterLoop (FILE *out, FILE *err);

};

} // namespace lldb_private

#endif // #ifndef liblldb_ScriptInterpreterNone_h_
