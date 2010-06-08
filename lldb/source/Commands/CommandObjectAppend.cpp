//===-- CommandObjectAppend.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectAppend.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-----------------------------------------------------------------------------
// CommandObjectAppend
//-----------------------------------------------------------------------------

CommandObjectAppend::CommandObjectAppend () :
    CommandObject ("append",
                   "Allows the user to append a value to a single debugger setting variable, for settings that are of list types. Type 'settings' to see a list of debugger setting variables",
                   "append <var-name> <value-string>")
{
}

CommandObjectAppend::~CommandObjectAppend ()
{
}

bool
CommandObjectAppend::Execute
(
    Args& command,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    CommandInterpreter::VariableMap::iterator pos;

    const int argc = command.GetArgumentCount();
    if (argc < 2)
    {
        result.AppendError ("'append' requires at least two arguments");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    const char *var_name = command.GetArgumentAtIndex(0);
    command.Shift();


    if (var_name == NULL || var_name[0] == '\0')
    {
        result.AppendError ("'set' command requires a valid variable name. No value supplied");
        result.SetStatus (eReturnStatusFailed);
    }
    else
    {
        StateVariable *var = interpreter->GetStateVariable(var_name);
        if (var == NULL)
        {
            result.AppendErrorWithFormat ("'%s' is not a settable internal variable.\n", var_name);
            result.SetStatus (eReturnStatusFailed);
        }
        else
        {
            if (var->GetType() == StateVariable::eTypeString)
            {
                for (int i = 0; i < command.GetArgumentCount(); ++i)
                    var->AppendStringValue (command.GetArgumentAtIndex(i));
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
            else if (var->GetType() == StateVariable::eTypeStringArray)
            {
                var->GetArgs().AppendArguments (command);
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
            else
            {
                result.AppendErrorWithFormat ("Values cannot be appended to variable '%s'.  Try 'set' instead.\n", var_name);
                result.SetStatus (eReturnStatusFailed);
            }
        }
    }
    return result.Succeeded();
}

