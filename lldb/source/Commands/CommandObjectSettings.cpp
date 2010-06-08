//===-- CommandObjectSettings.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectSettings.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectSettings
//-------------------------------------------------------------------------

CommandObjectSettings::CommandObjectSettings () :
    CommandObject ("settings",
                   "Lists the debugger settings variables available to the user to 'set' or 'show'.",
                   "settings")
{
}

CommandObjectSettings::~CommandObjectSettings()
{
}


bool
CommandObjectSettings::Execute
(
    Args& command,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    CommandInterpreter::VariableMap::iterator pos;

    if (command.GetArgumentCount() != 0)
    {
        result.AppendError ("'settings' does not take any arguments");
        result.SetStatus (eReturnStatusFailed);
    }
    else
    {
        interpreter->ShowVariableHelp (result);
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
    }

    return result.Succeeded();
}

