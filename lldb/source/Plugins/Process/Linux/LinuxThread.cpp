//===-- LinuxThread.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Host/Host.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"

#include "LinuxStopInfo.h"
#include "LinuxThread.h"
#include "ProcessLinux.h"
#include "ProcessMonitor.h"
#include "RegisterContextLinux_i386.h"
#include "RegisterContextLinux_x86_64.h"
#include "UnwindLLDB.h"

using namespace lldb_private;


LinuxThread::LinuxThread(Process &process, lldb::tid_t tid)
    : Thread(process, tid),
      m_frame_ap(0)
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
}

const char *
LinuxThread::GetInfo()
{
    return NULL;
}

lldb::RegisterContextSP
LinuxThread::GetRegisterContext()
{
    if (!m_reg_context_sp)
    {
        ArchSpec arch = Host::GetArchitecture();

        switch (arch.GetCore())
        {
        default:
            assert(false && "CPU type not supported!");
            break;

        case ArchSpec::eCore_x86_32_i386:
        case ArchSpec::eCore_x86_32_i486:
        case ArchSpec::eCore_x86_32_i486sx:
            m_reg_context_sp.reset(new RegisterContextLinux_i386(*this, 0));
            break;

        case ArchSpec::eCore_x86_64_x86_64:
            m_reg_context_sp.reset(new RegisterContextLinux_x86_64(*this, 0));
            break;
        }
    }
    return m_reg_context_sp;
}

lldb::RegisterContextSP
LinuxThread::CreateRegisterContextForFrame(lldb_private::StackFrame *frame)
{
    lldb::RegisterContextSP reg_ctx_sp;
    uint32_t concrete_frame_idx = 0;

    if (frame)
        concrete_frame_idx = frame->GetConcreteFrameIndex();

    if (concrete_frame_idx == 0)
        reg_ctx_sp = GetRegisterContext();
    else
        reg_ctx_sp = GetUnwinder()->CreateRegisterContextForFrame(frame);

    return reg_ctx_sp;
}

lldb::StopInfoSP
LinuxThread::GetPrivateStopReason()
{
    return m_stop_info;
}

Unwind *
LinuxThread::GetUnwinder()
{
    if (m_unwinder_ap.get() == NULL)
        m_unwinder_ap.reset(new UnwindLLDB(*this));

    return m_unwinder_ap.get();
}

bool
LinuxThread::WillResume(lldb::StateType resume_state)
{
    SetResumeState(resume_state);

    ClearStackFrames();
    if (m_unwinder_ap.get())
        m_unwinder_ap->Clear();

    return Thread::WillResume(resume_state);
}

bool
LinuxThread::Resume()
{
    lldb::StateType resume_state = GetResumeState();
    ProcessMonitor &monitor = GetMonitor();
    bool status;

    switch (resume_state)
    {
    default:
        assert(false && "Unexpected state for resume!");
        status = false;
        break;

    case lldb::eStateRunning:
        SetState(resume_state);
        status = monitor.Resume(GetID(), GetResumeSignal());
        break;

    case lldb::eStateStepping:
        SetState(resume_state);
        status = monitor.SingleStep(GetID(), GetResumeSignal());
        break;
    }

    return status;
}

void
LinuxThread::Notify(const ProcessMessage &message)
{
    switch (message.GetKind())
    {
    default:
        assert(false && "Unexpected message kind!");
        break;

    case ProcessMessage::eLimboMessage:
        LimboNotify(message);
        break;
        
    case ProcessMessage::eSignalMessage:
        SignalNotify(message);
        break;

    case ProcessMessage::eSignalDeliveredMessage:
        SignalDeliveredNotify(message);
        break;

    case ProcessMessage::eTraceMessage:
        TraceNotify(message);
        break;

    case ProcessMessage::eBreakpointMessage:
        BreakNotify(message);
        break;

    case ProcessMessage::eCrashMessage:
        CrashNotify(message);
        break;
    }
}

void
LinuxThread::BreakNotify(const ProcessMessage &message)
{
    bool status;

    status = GetRegisterContextLinux()->UpdateAfterBreakpoint();
    assert(status && "Breakpoint update failed!");

    // With our register state restored, resolve the breakpoint object
    // corresponding to our current PC.
    lldb::addr_t pc = GetRegisterContext()->GetPC();
    lldb::BreakpointSiteSP bp_site(GetProcess().GetBreakpointSiteList().FindByAddress(pc));
    lldb::break_id_t bp_id = bp_site->GetID();
    assert(bp_site && bp_site->ValidForThisThread(this));

    
    m_breakpoint = bp_site;
    m_stop_info = StopInfo::CreateStopReasonWithBreakpointSiteID(*this, bp_id);
}

void
LinuxThread::TraceNotify(const ProcessMessage &message)
{
    m_stop_info = StopInfo::CreateStopReasonToTrace(*this);
}

void
LinuxThread::LimboNotify(const ProcessMessage &message)
{
    m_stop_info = lldb::StopInfoSP(new LinuxLimboStopInfo(*this));
}

void
LinuxThread::SignalNotify(const ProcessMessage &message)
{
    int signo = message.GetSignal();

    m_stop_info = StopInfo::CreateStopReasonWithSignal(*this, signo);
    SetResumeSignal(signo);
}

void
LinuxThread::SignalDeliveredNotify(const ProcessMessage &message)
{
    int signo = message.GetSignal();

    // Just treat debugger generated signal events like breakpoints for now.
    m_stop_info = StopInfo::CreateStopReasonToTrace(*this);
    SetResumeSignal(signo);
}

void
LinuxThread::CrashNotify(const ProcessMessage &message)
{
    int signo = message.GetSignal();

    assert(message.GetKind() == ProcessMessage::eCrashMessage);

    m_stop_info = lldb::StopInfoSP(new LinuxCrashStopInfo(
                                       *this, signo, message.GetCrashReason()));
    SetResumeSignal(signo);
}
