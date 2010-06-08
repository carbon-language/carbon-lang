//===-- CommandObjectStatus.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectStatus.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "CommandObjectThread.h"

#include "lldb/Core/State.h"

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectStatus
//-------------------------------------------------------------------------

CommandObjectStatus::CommandObjectStatus () :
    CommandObject ("status",
                   "Shows the current status and location of executing process.",
                   "status",
                   0)
{
}

CommandObjectStatus::~CommandObjectStatus()
{
}


bool
CommandObjectStatus::Execute
(
    Args& command,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    StreamString &output_stream = result.GetOutputStream();
    result.SetStatus (eReturnStatusSuccessFinishNoResult);
    ExecutionContext exe_ctx(context->GetExecutionContext());
    if (exe_ctx.process)
    {
        const StateType state = exe_ctx.process->GetState();
        if (StateIsStoppedState(state))
        {
            if (state == eStateExited)
            {
                int exit_status = exe_ctx.process->GetExitStatus();
                const char *exit_description = exe_ctx.process->GetExitDescription();
                output_stream.Printf ("Process %d exited with status = %i (0x%8.8x) %s\n",
                                      exe_ctx.process->GetID(),
                                      exit_status,
                                      exit_status,
                                      exit_description ? exit_description : "");
            }
            else
            {
                output_stream.Printf ("Process %d %s\n", exe_ctx.process->GetID(), StateAsCString (state));
                if (exe_ctx.thread == NULL)
                    exe_ctx.thread = exe_ctx.process->GetThreadList().GetThreadAtIndex(0).get();
                if (exe_ctx.thread != NULL)
                {
                    DisplayThreadsInfo (interpreter, &exe_ctx, result, true, true);
                }
                else
                {
                    result.AppendError ("No valid thread found in current process.");
                    result.SetStatus (eReturnStatusFailed);
                }
            }
        }
    }
    else
    {
        result.AppendError ("No current location or status available.");
        result.SetStatus (eReturnStatusFailed);
    }
    return result.Succeeded();
}

