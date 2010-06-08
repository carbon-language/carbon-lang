//===-- CommandObjectRemove.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectRemove.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectRemove
//-------------------------------------------------------------------------

CommandObjectRemove::CommandObjectRemove () :
    CommandObject ("remove",
                     "Allows the user to remove/delete user-defined command functions (script functions).",
                     "remove <command-name-to-be-removed>")
{
}

CommandObjectRemove::~CommandObjectRemove()
{
}


bool
CommandObjectRemove::Execute (Args& args, CommandContext *context, CommandInterpreter *interpreter,
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

                if (interpreter->RemoveUser (command_name) == false)
                {
                    if (interpreter->UserCommandExists (command_name))
                        result.AppendErrorWithFormat ("Unknown error occurred; unable to remove command '%s'.\n",
                                                     command_name);
                    else
                        result.AppendErrorWithFormat ("'%s' is not a user-defined command/function name.\n",
                                                     command_name);
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
        result.AppendError ("must call remove with a valid command");
        result.SetStatus (eReturnStatusFailed);
    }

    return result.Succeeded();
}

