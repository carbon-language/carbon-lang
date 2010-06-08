//===-- CommandObjectShow.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectShow.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectShow
//-------------------------------------------------------------------------

CommandObjectShow::CommandObjectShow () :
    CommandObject ("show",
                    "Allows the user to see a single debugger setting variable and its value, or lists them all.",
                    "show [<setting-variable-name>]")
{
}

CommandObjectShow::~CommandObjectShow()
{
}


bool
CommandObjectShow::Execute
(
    Args& command,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    CommandInterpreter::VariableMap::iterator pos;

    if (command.GetArgumentCount())
    {
        // The user requested to see the value of a particular variable.

        const char *var_name = command.GetArgumentAtIndex(0);
        StateVariable *var = interpreter->GetStateVariable(var_name);
        if (var)
        {
            var->AppendVariableInformation (result);
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendErrorWithFormat ("Unrecognized variable '%s'; cannot do 'show' command.\n", var_name);
            result.SetStatus (eReturnStatusFailed);
        }
    }
    else
    {
        // The user didn't specify a particular variable, so show the values of all of them.
        interpreter->ShowVariableValues(result);
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
    }

    return result.Succeeded();
}
