//===-- CommandObjectQuit.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectQuit.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "CommandInterpreter.h"
#include "CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectQuit
//-------------------------------------------------------------------------

CommandObjectQuit::CommandObjectQuit () :
    CommandObject ("quit", "Quits out of the LLDB debugger.", "quit")
{
}

CommandObjectQuit::~CommandObjectQuit ()
{
}

bool
CommandObjectQuit::Execute
(
    Args& command,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    interpreter->BroadcastEvent (CommandInterpreter::eBroadcastBitQuitCommandReceived);
    result.SetStatus (eReturnStatusQuit);
    return true;
}

