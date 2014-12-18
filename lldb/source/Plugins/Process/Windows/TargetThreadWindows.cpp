//===-- TargetThreadWindows.cpp----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostNativeThreadBase.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Target/RegisterContext.h"

#include "TargetThreadWindows.h"
#include "ProcessWindows.h"
#include "UnwindLLDB.h"

#if !defined(_WIN64)
#include "x86/RegisterContextWindows_x86.h"
#endif

using namespace lldb;
using namespace lldb_private;

TargetThreadWindows::TargetThreadWindows(ProcessWindows &process, const HostThread &thread)
    : Thread(process, thread.GetNativeThread().GetThreadId())
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
    GetRegisterContext()->InvalidateIfNeeded(false);
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
    if (!m_reg_context_sp)
        m_reg_context_sp = CreateRegisterContextForFrameIndex(0);

    return m_reg_context_sp;
}

RegisterContextSP
TargetThreadWindows::CreateRegisterContextForFrame(StackFrame *frame)
{
    return CreateRegisterContextForFrameIndex(frame->GetConcreteFrameIndex());
}

RegisterContextSP
TargetThreadWindows::CreateRegisterContextForFrameIndex(uint32_t idx)
{
    if (!m_reg_context_sp)
    {
        ArchSpec arch = HostInfo::GetArchitecture();
        switch (arch.GetMachine())
        {
            case llvm::Triple::x86:
#if defined(_WIN64)
                // FIXME: This is a Wow64 process, create a RegisterContextWindows_Wow64
#else
                m_reg_context_sp.reset(new RegisterContextWindows_x86(*this, idx));
#endif
                break;
            case llvm::Triple::x86_64:
#if defined(_WIN64)
                // FIXME: This is a 64-bit process, create a RegisterContextWindows_x86_64
#else
                // LLDB is 32-bit, but the target process is 64-bit.  We probably can't debug this.
#endif
            default:
                break;
        }
    }
    return m_reg_context_sp;
}

bool
TargetThreadWindows::CalculateStopInfo()
{
    SetStopInfo(m_stop_info_sp);
    return true;
}

Unwind *
TargetThreadWindows::GetUnwinder()
{
    // FIXME: Implement an unwinder based on the Windows unwinder exposed through DIA SDK.
    if (m_unwinder_ap.get() == NULL)
        m_unwinder_ap.reset(new UnwindLLDB(*this));
    return m_unwinder_ap.get();
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
