//===-- CommandContext.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandContext.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

CommandContext::CommandContext () :
    m_exe_ctx ()
{
}

CommandContext::~CommandContext ()
{
}

Target *
CommandContext::GetTarget()
{
    return Debugger::GetSharedInstance().GetCurrentTarget().get();
}


ExecutionContext &
CommandContext::GetExecutionContext()
{
    return m_exe_ctx;
}

void
CommandContext::Update (ExecutionContext *override_context)
{
    m_exe_ctx.Clear();

    if (override_context != NULL)
    {
        m_exe_ctx.target = override_context->target;
        m_exe_ctx.process = override_context->process;
        m_exe_ctx.thread = override_context->thread;
        m_exe_ctx.frame = override_context->frame;
    }
    else
    {
        TargetSP target_sp (Debugger::GetSharedInstance().GetCurrentTarget());
        if (target_sp)
        {
            m_exe_ctx.process = target_sp->GetProcessSP().get();
            if (m_exe_ctx.process && m_exe_ctx.process->IsRunning() == false)
            {
                m_exe_ctx.thread = m_exe_ctx.process->GetThreadList().GetCurrentThread().get();
                if (m_exe_ctx.thread == NULL)
                    m_exe_ctx.thread = m_exe_ctx.process->GetThreadList().GetThreadAtIndex(0).get();
                if (m_exe_ctx.thread)
                {
                    m_exe_ctx.frame = m_exe_ctx.thread->GetCurrentFrame().get();
                    if (m_exe_ctx.frame == NULL)
                        m_exe_ctx.frame = m_exe_ctx.thread->GetStackFrameAtIndex (0).get();
                }
            }
        }
    }
}
