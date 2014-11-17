//===-- TargetThreadWindows.cpp----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TargetThreadWindows.h"
#include "ProcessWindows.h"
#include "lldb/Host/HostNativeThreadBase.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"

using namespace lldb;
using namespace lldb_private;

TargetThreadWindows::TargetThreadWindows(ProcessWindows &process, const HostThread &thread)
    : Thread(process, ((HostThreadWindows &)thread.GetNativeThread()).GetThreadId())
    , m_host_thread(thread)
{
}

TargetThreadWindows::~TargetThreadWindows()
{
    DestroyThread();
}

void
TargetThreadWindows::RefreshStateAfterStop()
{
}

void
TargetThreadWindows::WillResume(lldb::StateType resume_state)
{
}

void
TargetThreadWindows::DidStop()
{
}

RegisterContextSP
TargetThreadWindows::GetRegisterContext()
{
    return RegisterContextSP();
}

RegisterContextSP
TargetThreadWindows::CreateRegisterContextForFrame(StackFrame *frame)
{
    return RegisterContextSP();
}

bool
TargetThreadWindows::CalculateStopInfo()
{
    return false;
}

bool
TargetThreadWindows::DoResume()
{
    StateType resume_state = GetResumeState();
    StateType current_state = GetState();
    if (resume_state == current_state)
        return true;

    bool success = false;
    DWORD suspend_count = 0;
    switch (resume_state)
    {
        case eStateRunning:
            SetState(resume_state);
            do
            {
                suspend_count = ::ResumeThread(m_host_thread.GetNativeThread().GetSystemHandle());
            } while (suspend_count > 1 && suspend_count != (DWORD)-1);
            success = (suspend_count != (DWORD)-1);
            break;
        case eStateStopped:
        case eStateSuspended:
            if (current_state != eStateStopped && current_state != eStateSuspended)
            {
                suspend_count = SuspendThread(m_host_thread.GetNativeThread().GetSystemHandle());
                success = (suspend_count != (DWORD)-1);
            }
            break;
        default:
            success = false;
    }
    return success;
}
