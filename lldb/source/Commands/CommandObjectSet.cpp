//===-- CommandObjectSet.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectSet.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectSet
//-------------------------------------------------------------------------

CommandObjectSet::CommandObjectSet () :
    CommandObject ("set",
                   "Allows the user to set or change the value of a single debugger setting variable.",
                   "set <setting_name> <value>")
{
}

CommandObjectSet::~CommandObjectSet()
{
}


bool
CommandObjectSet::Execute
(
    Args& command,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    CommandInterpreter::VariableMap::iterator pos;

    const int argc = command.GetArgumentCount();

    if (argc < 1)
    {
        result.AppendError ("'set' takes at least two arguments");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    const char *var_name = command.GetArgumentAtIndex(0);
    const char *var_value = command.GetArgumentAtIndex(1);

    if (var_name == NULL || var_name[0] == '\0')
    {
        result.AppendError ("'set' command requires a valid variable name; No value supplied");
        result.SetStatus (eReturnStatusFailed);
    }
    else if (var_value == NULL || var_value[0] == '\0')
    {
        // No value given:  Check to see if we're trying to clear an array.
        StateVariable *var = interpreter->GetStateVariable (var_name);
        if (var != NULL
            && var->GetType() == StateVariable::eTypeStringArray)
        {
            var->ArrayClearValues();
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError ("'set' command requires a valid variable value; No value supplied");
            result.SetStatus (eReturnStatusFailed);
        }
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
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
            if (var->GetType() == StateVariable::eTypeBoolean)
            {
                bool success = false;
                bool new_value = Args::StringToBoolean (var_value, false, &success);

                if (success)
                {
                    result.SetStatus(eReturnStatusSuccessFinishResult);
                    if (!var->HasVerifyFunction() || var->VerifyValue (interpreter, (void *) &new_value, result))
                        var->SetBoolValue (new_value);
                }
                else
                {
                    result.AppendErrorWithFormat ("Invalid boolean string '%s'.\n", var_value);
                    result.SetStatus (eReturnStatusFailed);
                }
            }
            else if (var->GetType() == StateVariable::eTypeInteger)
            {
                bool success = false;
                int new_value = Args::StringToSInt32(var_value, -1, 0, &success);

                if (success)
                {
                    result.SetStatus(eReturnStatusSuccessFinishResult);
                    if (!var->HasVerifyFunction() || var->VerifyValue (interpreter, (void *) &new_value, result))
                        var->SetIntValue (new_value);
                }
                else
                {
                    result.AppendErrorWithFormat ("Invalid boolean string '%s'.\n", var_value);
                    result.SetStatus (eReturnStatusFailed);
                }
            }
            else if (var->GetType() == StateVariable::eTypeString)
            {
                if (!var->HasVerifyFunction() || var->VerifyValue (interpreter, (void *) var_value, result))
                    var->SetStringValue (var_value);
            }
            else if (var->GetType() == StateVariable::eTypeStringArray)
            {
                if (var_value == NULL || var_value[0] == '\0')
                    var->ArrayClearValues ();
                else
                {
                    command.Shift(); // shift off variable name
                    var->ArrayClearValues(); // clear the old values
                    var->GetArgs().AppendArguments (command); // set the new values.
                }
            }
            else
            {
                result.AppendErrorWithFormat ("Variable '%s' has unrecognized type.\n",
                                              var->GetName());
                result.SetStatus (eReturnStatusFailed);
            }
        }
    }
    return result.Succeeded();
}

