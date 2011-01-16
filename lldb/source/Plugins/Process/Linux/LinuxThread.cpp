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
      m_stop_info_id(0),
      m_note(eNone)
{
}

LinuxThread::~LinuxThread()
{
    DestroyThread();
}

ProcessMonitor &
LinuxThread::GetMonitor()
{
    ProcessLinux &process = static_cast<ProcessLinux&>(GetProcess());
    return process.GetMonitor();
}

void
LinuxThread::RefreshStateAfterStop()
{
    RefreshPrivateStopReason();
}

const char *
LinuxThread::GetInfo()
{
    return NULL;
}

lldb::RegisterContextSP
LinuxThread::GetRegisterContext()
{
    ProcessLinux &process = static_cast<ProcessLinux&>(GetProcess());

    if (!m_reg_context_sp)
    {
        ArchSpec arch = process.GetTarget().GetArchitecture();

        switch (arch.GetGenericCPUType())
        {
        default:
            assert(false && "CPU type not supported!");
            break;

        case ArchSpec::eCPU_x86_64:
            m_reg_context_sp.reset(new RegisterContextLinux_x86_64(*this, 0));
            break;
        }
    }
    return m_reg_context_sp;
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

lldb::RegisterContextSP
LinuxThread::CreateRegisterContextForFrame (lldb_private::StackFrame *frame)
{
    lldb::RegisterContextSP reg_ctx_sp;
    uint32_t concrete_frame_idx = 0;
    if (frame)
        concrete_frame_idx = frame->GetConcreteFrameIndex();
        
    if (concrete_frame_idx == 0)
        reg_ctx_sp = GetRegisterContext();
    else
        reg_ctx_sp.reset (new RegisterContextLinux_x86_64(*this, frame->GetConcreteFrameIndex()));
    return reg_ctx_sp;
}

lldb::StopInfoSP
LinuxThread::GetPrivateStopReason()
{
    const uint32_t process_stop_id = GetProcess().GetStopID();

    if (m_stop_info_id != process_stop_id || !m_stop_info || !m_stop_info->IsValid())
        RefreshPrivateStopReason();
    return m_stop_info;
}

Unwind *
LinuxThread::GetUnwinder()
{
    return m_unwinder_ap.get();
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

    status = GetRegisterContextLinux()->UpdateAfterBreakpoint();
    assert(status && "Breakpoint update failed!");

    // With our register state restored, resolve the breakpoint object
    // corresponding to our current PC.
    lldb::addr_t pc = GetRegisterContext()->GetPC();
    lldb::BreakpointSiteSP bp_site(GetProcess().GetBreakpointSiteList().FindByAddress(pc));
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

void
LinuxThread::RefreshPrivateStopReason()
{
    m_stop_info_id = GetProcess().GetStopID();

    switch (m_note) {

    default:
    case eNone:
        m_stop_info.reset();
        break;

    case eBreak:
        m_stop_info = StopInfo::CreateStopReasonWithBreakpointSiteID(
            *this, m_breakpoint->GetID());
        break;

    case eTrace:
        m_stop_info = StopInfo::CreateStopReasonToTrace(*this);
        break;

    case eExit:
        m_stop_info = StopInfo::CreateStopReasonWithSignal(*this, SIGCHLD);
        break;
    }
}
