//===-- ExecutionContext.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//

//
//===----------------------------------------------------------------------===//


#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb_private;

ExecutionContext::ExecutionContext() :
    target (NULL),
    process (NULL),
    thread (NULL),
    frame (NULL)
{
}

ExecutionContext::ExecutionContext (Target* t, bool fill_current_process_thread_frame) :
    target (t),
    process (NULL),
    thread (NULL),
    frame (NULL)
{
    if (t && fill_current_process_thread_frame)
    {
        process = t->GetProcessSP().get();
        if (process)
        {
            thread = process->GetThreadList().GetSelectedThread().get();
            if (thread)
                frame = thread->GetSelectedFrame().get();
        }
    }
}

ExecutionContext::ExecutionContext(Process* p, Thread *t, StackFrame *f) :
    target (p ? &p->GetTarget() : NULL),
    process (p),
    thread (t),
    frame (f)
{
}

ExecutionContext::ExecutionContext (ExecutionContextScope *exe_scope_ptr)
{
    if (exe_scope_ptr)
        exe_scope_ptr->Calculate (*this);
    else
    {
        target  = NULL;
        process = NULL;
        thread  = NULL;
        frame   = NULL;
    }
}

ExecutionContext::ExecutionContext (ExecutionContextScope &exe_scope_ref)
{
    exe_scope_ref.Calculate (*this);
}

void
ExecutionContext::Clear()
{
    target  = NULL;
    process = NULL;
    thread  = NULL;
    frame   = NULL;
}


RegisterContext *
ExecutionContext::GetRegisterContext () const
{
    if (frame)
        return frame->GetRegisterContext();
    else if (thread)
        return thread->GetRegisterContext();
    return NULL;
}

ExecutionContextScope *
ExecutionContext::GetBestExecutionContextScope () const
{
    if (frame)
        return frame;
    if (thread)
        return thread;
    if (process)
        return process;
    return target;
}
