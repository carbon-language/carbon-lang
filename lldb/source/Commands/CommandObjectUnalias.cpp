//===-- CommandObjectUnalias.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectUnalias.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectUnalias
//-------------------------------------------------------------------------

CommandObjectUnalias::CommandObjectUnalias () :
    CommandObject ("unalias",
                     "Allows the user to remove/delete a user-defined command abbreviation.",
                     "unalias <alias-name-to-be-removed>")
{
}

CommandObjectUnalias::~CommandObjectUnalias()
{
}


bool
CommandObjectUnalias::Execute (Args& args, CommandContext *context, CommandInterpreter *interpreter,
                               CommandReturnObject &result)
{
    CommandObject::CommandMap::iterator pos;
    CommandObject *cmd_obj;

    if (args.GetArgumentCount() != 0)
    {
        const char *command_name = args.GetArgumentAtIndex(0);
        cmd_obj = interpreter->GetCommandObject(command_name);
        if (cmd_obj)
        {
            if (interpreter->CommandExists (command_name))
            {
                result.AppendErrorWithFormat ("'%s' is a permanent debugger command and cannot be removed.\n",
                                              command_name);
                result.SetStatus (eReturnStatusFailed);
            }
            else
            {

                if (interpreter->RemoveAlias (command_name) == false)
                {
                    if (interpreter->AliasExists (command_name))
                        result.AppendErrorWithFormat ("Error occurred while attempting to unalias '%s'.\n", command_name);
                    else
                        result.AppendErrorWithFormat ("'%s' is not an existing alias.\n", command_name);
                    result.SetStatus (eReturnStatusFailed);
                }
                else
                    result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
        }
        else
        {
            result.AppendErrorWithFormat ("'%s' is not a known command.\nTry 'help' to see a current list of commands.\n",
                                         command_name);
            result.SetStatus (eReturnStatusFailed);
        }
    }
    else
    {
        result.AppendError ("must call 'unalias' with a valid alias");
        result.SetStatus (eReturnStatusFailed);
    }

    return result.Succeeded();
}

