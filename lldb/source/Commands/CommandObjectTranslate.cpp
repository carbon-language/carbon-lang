//===-- CommandObjectTranslate.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectTranslate.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Args.h"
#include "lldb/Core/Options.h"

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectTranslate
//-------------------------------------------------------------------------

CommandObjectTranslate::CommandObjectTranslate () :
    CommandObject ("translate",
                     "Shows the actual function called for a given debugger command.",
                     "translate <command>")
{
}

CommandObjectTranslate::~CommandObjectTranslate()
{
}


bool
CommandObjectTranslate::Execute
(
    Args& command,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    CommandObject *cmd_obj;

    if (command.GetArgumentCount() != 0)
    {
        cmd_obj = interpreter->GetCommandObject(command.GetArgumentAtIndex(0));
        if (cmd_obj)
        {
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
            result.AppendMessageWithFormat ("%s\n", cmd_obj->Translate());
        }
        else
        {
            result.AppendErrroWithFormat
            ("'%s' is not a known command.\nTry 'help' to see a current list of commands.\n",
             command.GetArgumentAtIndex(0));
            result.SetStatus (eReturnStatusFailed);
        }
    }
    else
    {
        result.AppendError ("must call translate with a valid command");
        result.SetStatus (eReturnStatusFailed);
    }

    return result.Succeeded();
}
