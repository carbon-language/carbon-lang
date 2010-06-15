//===-- CommandObjectAdd.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectAdd.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectAdd
//-------------------------------------------------------------------------

CommandObjectAdd::CommandObjectAdd () :
    CommandObject ("add",
                     "Allows the user to add a new command/function pair to the debugger's dictionary.",
                     "add <new-command-name> <script-function-name>")
{
}

CommandObjectAdd::~CommandObjectAdd()
{
}


bool
CommandObjectAdd::Execute
(
    Args& command,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    result.AppendMessage ("This function has not been implemented yet.");
    result.SetStatus (eReturnStatusSuccessFinishNoResult);
    return result.Succeeded();
}
