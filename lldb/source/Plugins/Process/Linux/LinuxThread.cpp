//===-- LinuxThread.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <errno.h>

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Target/Process.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"

#include "LinuxThread.h"
#include "ProcessLinux.h"
#include "ProcessMonitor.h"
#include "RegisterContextLinux_x86_64.h"

using namespace lldb_private;

LinuxThread::LinuxThread(Process &process, lldb::tid_t tid)
    : Thread(process, tid),
      m_frame_ap(0),
      m_register_ap(0),
      m_note(eNone)
{
    ArchSpec arch = process.GetTarget().GetArchitecture();

    switch (arch.GetGenericCPUType())
    {
    default:
        assert(false && "CPU type not supported!");
        break;

    case ArchSpec::eCPU_x86_64:
        m_register_ap.reset(new RegisterContextLinux_x86_64(*this, NULL));
        break;
    }
}

ProcessMonitor &
LinuxThread::GetMonitor()
{
    ProcessLinux *process = static_cast<ProcessLinux*>(CalculateProcess());
    return process->GetMonitor();
}

void
LinuxThread::RefreshStateAfterStop()
{
}

const char *
LinuxThread::GetInfo()
{
    return NULL;
}

RegisterContextLinux *
LinuxThread::GetRegisterContext()
{
    return m_register_ap.get();
}

bool
LinuxThread::SaveFrameZeroState(RegisterCheckpoint &checkpoint)
{
    return false;
}

bool
LinuxThread::RestoreSaveFrameZero(const RegisterCheckpoint &checkpoint)
{
    return false;
}

RegisterContextLinux *
LinuxThread::CreateRegisterContextForFrame(lldb_private::StackFrame *frame)
{
    return new RegisterContextLinux_x86_64(*this, frame);
}

bool
LinuxThread::GetRawStopReason(StopInfo *stop_info)
{
    stop_info->Clear();

    switch (m_note)
    {
    default:
        stop_info->SetStopReasonToNone();
        break;

    case eBreak:
        stop_info->SetStopReasonWithBreakpointSiteID(m_breakpoint->GetID());
        break;

    case eTrace:
        stop_info->SetStopReasonToTrace();
    }

    return true;
}

bool
LinuxThread::WillResume(lldb::StateType resume_state)
{
    SetResumeState(resume_state);
    return Thread::WillResume(resume_state);
}

bool
LinuxThread::Resume()
{
    lldb::StateType resume_state = GetResumeState();
    ProcessMonitor &monitor = GetMonitor();
    bool status;

    switch (GetResumeState())
    {
    default:
        assert(false && "Unexpected state for resume!");
        status = false;
        break;

    case lldb::eStateSuspended:
        // FIXME: Implement process suspension.
        status = false;

    case lldb::eStateRunning:
        SetState(resume_state);
        status = monitor.Resume(GetID());
        break;

    case lldb::eStateStepping:
        SetState(resume_state);
        status = GetRegisterContext()->HardwareSingleStep(true);
        break;
    }

    m_note = eNone;
    return status;
}

void
LinuxThread::BreakNotify()
{
    bool status;

    status = GetRegisterContext()->UpdateAfterBreakpoint();
    assert(status && "Breakpoint update failed!");

    // With our register state restored, resolve the breakpoint object
    // corresponding to our current PC.
    lldb::addr_t pc = GetRegisterContext()->GetPC();
    lldb::BreakpointSiteSP bp_site =
        GetProcess().GetBreakpointSiteList().FindByAddress(pc);
    assert(bp_site && bp_site->ValidForThisThread(this));

    m_note = eBreak;
    m_breakpoint = bp_site;
}

void
LinuxThread::TraceNotify()
{
    m_note = eTrace;
}

void
LinuxThread::ExitNotify()
{
    m_note = eExit;
}
