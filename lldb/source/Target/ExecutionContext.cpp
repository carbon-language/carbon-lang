//===-- ExecutionContext.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
    m_target_sp (),
    m_process_sp (),
    m_thread_sp (),
    m_frame_sp ()
{
}

ExecutionContext::ExecutionContext (const ExecutionContext &rhs) :
    m_target_sp (rhs.m_target_sp),
    m_process_sp(rhs.m_process_sp),
    m_thread_sp (rhs.m_thread_sp),
    m_frame_sp  (rhs.m_frame_sp)
{
}

ExecutionContext &
ExecutionContext::operator =(const ExecutionContext &rhs)
{
    if (this != &rhs)
    {
        m_target_sp  = rhs.m_target_sp;
        m_process_sp = rhs.m_process_sp;
        m_thread_sp  = rhs.m_thread_sp;
        m_frame_sp   = rhs.m_frame_sp;
    }
    return *this;
}

ExecutionContext::ExecutionContext (Target* t, bool fill_current_process_thread_frame) :
    m_target_sp (t),
    m_process_sp (),
    m_thread_sp (),
    m_frame_sp ()
{
    if (t && fill_current_process_thread_frame)
    {
        m_process_sp = t->GetProcessSP();
        if (m_process_sp)
        {
            m_thread_sp = m_process_sp->GetThreadList().GetSelectedThread();
            if (m_thread_sp)
                m_frame_sp = m_thread_sp->GetSelectedFrame().get();
        }
    }
}

ExecutionContext::ExecutionContext(Process* process, Thread *thread, StackFrame *frame) :
    m_target_sp (process ? &process->GetTarget() : NULL),
    m_process_sp (process),
    m_thread_sp (thread),
    m_frame_sp (frame)
{
}

ExecutionContext::ExecutionContext (ExecutionContextScope *exe_scope_ptr)
{
    if (exe_scope_ptr)
        exe_scope_ptr->CalculateExecutionContext (*this);
    else
    {
        m_target_sp.reset();
        m_process_sp.reset();
        m_thread_sp.reset();
        m_frame_sp.reset();
    }
}

ExecutionContext::ExecutionContext (ExecutionContextScope &exe_scope_ref)
{
    exe_scope_ref.CalculateExecutionContext (*this);
}

void
ExecutionContext::Clear()
{
    m_target_sp.reset();
    m_process_sp.reset();
    m_thread_sp.reset();
    m_frame_sp.reset();
}

ExecutionContext::~ExecutionContext()
{
}


RegisterContext *
ExecutionContext::GetRegisterContext () const
{
    if (m_frame_sp)
        return m_frame_sp->GetRegisterContext().get();
    else if (m_thread_sp)
        return m_thread_sp->GetRegisterContext().get();
    return NULL;
}

Target *
ExecutionContext::GetTargetPtr () const
{
    if (m_target_sp)
        return m_target_sp.get();
    if (m_process_sp)
        return &m_process_sp->GetTarget();
    return NULL;
}

Process *
ExecutionContext::GetProcessPtr () const
{
    if (m_process_sp)
        return m_process_sp.get();
    if (m_target_sp)
        return m_target_sp->GetProcessSP().get();
    return NULL;
}

ExecutionContextScope *
ExecutionContext::GetBestExecutionContextScope () const
{
    if (m_frame_sp)
        return m_frame_sp.get();
    if (m_thread_sp)
        return m_thread_sp.get();
    if (m_process_sp)
        return m_process_sp.get();
    return m_target_sp.get();
}

Target &
ExecutionContext::GetTargetRef () const
{
    assert (m_target_sp.get());
    return *m_target_sp;
}

Process &
ExecutionContext::GetProcessRef () const
{
    assert (m_process_sp.get());
    return *m_process_sp;
}

Thread &
ExecutionContext::GetThreadRef () const
{
    assert (m_thread_sp.get());
    return *m_thread_sp;
}

StackFrame &
ExecutionContext::GetFrameRef () const
{
    assert (m_frame_sp.get());
    return *m_frame_sp;
}

void
ExecutionContext::SetTargetSP (const lldb::TargetSP &target_sp)
{
    m_target_sp = target_sp;
}

void
ExecutionContext::SetProcessSP (const lldb::ProcessSP &process_sp)
{
    m_process_sp = process_sp;
}

void
ExecutionContext::SetThreadSP (const lldb::ThreadSP &thread_sp)
{
    m_thread_sp = thread_sp;
}

void
ExecutionContext::SetFrameSP (const lldb::StackFrameSP &frame_sp)
{
    m_frame_sp = frame_sp;
}

void
ExecutionContext::SetTargetPtr (Target* target)
{
    m_target_sp = target;
}

void
ExecutionContext::SetProcessPtr (Process *process)
{
    m_process_sp = process;
}

void
ExecutionContext::SetThreadPtr (Thread *thread)
{
    m_thread_sp = thread;
}

void
ExecutionContext::SetFramePtr (StackFrame *frame)
{
    m_frame_sp = frame;
}

