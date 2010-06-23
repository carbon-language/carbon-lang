//===-- CommandObjectScript.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectScript_h_
#define liblldb_CommandObjectScript_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectScript
//-------------------------------------------------------------------------

class CommandObjectScript : public CommandObject
{
public:

    CommandObjectScript (lldb::ScriptLanguage script_lang);

    virtual
    ~CommandObjectScript ();

    bool WantsRawCommandString();

    virtual bool
    ExecuteRawCommandString (CommandInterpreter &interpreter,
                             const char *command,
                             CommandReturnObject &result);

    virtual bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result);

    ScriptInterpreter *
    GetInterpreter (CommandInterpreter &interpreter);

private:
    lldb::ScriptLanguage m_script_lang;
    std::auto_ptr<ScriptInterpreter> m_interpreter_ap;
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectScript_h_
