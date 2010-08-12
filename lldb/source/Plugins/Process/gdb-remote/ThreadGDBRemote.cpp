//===-- ThreadGDBRemote.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "ThreadGDBRemote.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Unwind.h"
#include "lldb/Breakpoint/WatchpointLocation.h"

#include "LibUnwindRegisterContext.h"
#include "ProcessGDBRemote.h"
#include "ProcessGDBRemoteLog.h"
#include "Utility/StringExtractorGDBRemote.h"
#include "UnwindLibUnwind.h"
#include "UnwindMacOSXFrameBackchain.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Thread Registers
//----------------------------------------------------------------------

ThreadGDBRemote::ThreadGDBRemote (ProcessGDBRemote &process, lldb::tid_t tid) :
    Thread(process, tid),
    m_thread_name (),
    m_dispatch_queue_name (),
    m_thread_dispatch_qaddr (LLDB_INVALID_ADDRESS)
{
//    ProcessGDBRemoteLog::LogIf(GDBR_LOG_THREAD | GDBR_LOG_VERBOSE, "ThreadGDBRemote::ThreadGDBRemote ( pid = %i, tid = 0x%4.4x, )", m_process.GetID(), GetID());
    ProcessGDBRemoteLog::LogIf(GDBR_LOG_THREAD, "%p: ThreadGDBRemote::ThreadGDBRemote (pid = %i, tid = 0x%4.4x)", this, m_process.GetID(), GetID());
}

ThreadGDBRemote::~ThreadGDBRemote ()
{
    ProcessGDBRemoteLog::LogIf(GDBR_LOG_THREAD, "%p: ThreadGDBRemote::~ThreadGDBRemote (pid = %i, tid = 0x%4.4x)", this, m_process.GetID(), GetID());
}


const char *
ThreadGDBRemote::GetInfo ()
{
    return NULL;
}


const char *
ThreadGDBRemote::GetName ()
{
    if (m_thread_name.empty())
        return NULL;
    return m_thread_name.c_str();
}


const char *
ThreadGDBRemote::GetQueueName ()
{
    // Always re-fetch the dispatch queue name since it can change
    if (m_thread_dispatch_qaddr != 0 || m_thread_dispatch_qaddr != LLDB_INVALID_ADDRESS)
        return GetGDBProcess().GetDispatchQueueNameForThread (m_thread_dispatch_qaddr, m_dispatch_queue_name);
    return NULL;
}

bool
ThreadGDBRemote::WillResume (StateType resume_state)
{
    // TODO: cache for next time in case we can match things up??
    ClearStackFrames();
    int signo = GetResumeSignal();

    switch (resume_state)
    {
    case eStateSuspended:
    case eStateStopped:
        // Don't append anything for threads that should stay stopped.
        break;

    case eStateRunning:
        if (m_process.GetUnixSignals().SignalIsValid (signo))
            GetGDBProcess().m_continue_packet.Printf(";C%2.2x:%4.4x", signo, GetID());
        else
            GetGDBProcess().m_continue_packet.Printf(";c:%4.4x", GetID());
        break;

    case eStateStepping:
        if (m_process.GetUnixSignals().SignalIsValid (signo))
            GetGDBProcess().m_continue_packet.Printf(";S%2.2x:%4.4x", signo, GetID());
        else
            GetGDBProcess().m_continue_packet.Printf(";s:%4.4x", GetID());
        break;

    default:
        break;
    }
    Thread::WillResume(resume_state);
    return true;
}

void
ThreadGDBRemote::RefreshStateAfterStop()
{
    // Invalidate all registers in our register context
    GetRegisterContext()->Invalidate();
}

Unwind *
ThreadGDBRemote::GetUnwinder ()
{
    if (m_unwinder_ap.get() == NULL)
    {
        const ArchSpec target_arch (GetProcess().GetTarget().GetArchitecture ());
        if (target_arch == ArchSpec("x86_64") ||  target_arch == ArchSpec("i386"))
        {
            m_unwinder_ap.reset (new UnwindLibUnwind (*this, GetGDBProcess().GetLibUnwindAddressSpace()));
        }
        else
        {
            m_unwinder_ap.reset (new UnwindMacOSXFrameBackchain (*this));
        }
    }
    return m_unwinder_ap.get();
}

void
ThreadGDBRemote::ClearStackFrames ()
{
    Unwind *unwinder = GetUnwinder ();
    if (unwinder)
        unwinder->Clear();
    Thread::ClearStackFrames();
}


bool
ThreadGDBRemote::ThreadIDIsValid (lldb::tid_t thread)
{
    return thread != 0;
}

void
ThreadGDBRemote::Dump(Log *log, uint32_t index)
{
}


bool
ThreadGDBRemote::ShouldStop (bool &step_more)
{
    return true;
}
RegisterContext *
ThreadGDBRemote::GetRegisterContext ()
{
    if (m_reg_context_sp.get() == NULL)
        m_reg_context_sp.reset (CreateRegisterContextForFrame (NULL));
    return m_reg_context_sp.get();
}

RegisterContext *
ThreadGDBRemote::CreateRegisterContextForFrame (StackFrame *frame)
{
    const bool read_all_registers_at_once = false;
    uint32_t frame_idx = 0;
    
    if (frame)
        frame_idx = frame->GetID();

    if (frame_idx == 0)
        return new GDBRemoteRegisterContext (*this, frame, GetGDBProcess().m_register_info, read_all_registers_at_once);
    else if (m_unwinder_ap.get() && frame_idx < m_unwinder_ap->GetFrameCount())
        return m_unwinder_ap->CreateRegisterContextForFrame (frame);
    return NULL;
}

bool
ThreadGDBRemote::SaveFrameZeroState (RegisterCheckpoint &checkpoint)
{
    lldb::StackFrameSP frame_sp(GetStackFrameAtIndex (0));
    if (frame_sp)
    {
        checkpoint.SetStackID(frame_sp->GetStackID());
        return frame_sp->GetRegisterContext()->ReadAllRegisterValues (checkpoint.GetData());
    }
    return false;
}

bool
ThreadGDBRemote::RestoreSaveFrameZero (const RegisterCheckpoint &checkpoint)
{
    lldb::StackFrameSP frame_sp(GetStackFrameAtIndex (0));
    if (frame_sp)
    {
        bool ret = frame_sp->GetRegisterContext()->WriteAllRegisterValues (checkpoint.GetData());
        frame_sp->GetRegisterContext()->Invalidate();
        ClearStackFrames();
        return ret;
    }
    return false;
}

lldb::StopInfoSP
ThreadGDBRemote::GetPrivateStopReason ()
{
    if (m_actual_stop_info_sp.get() == NULL || m_actual_stop_info_sp->IsValid() == false)
    {
        m_actual_stop_info_sp.reset();

        char packet[256];
        ::snprintf(packet, sizeof(packet), "qThreadStopInfo%x", GetID());
        StringExtractorGDBRemote stop_packet;
        if (GetGDBProcess().GetGDBRemote().SendPacketAndWaitForResponse(packet, stop_packet, 1, false))
        {
            std::string copy(stop_packet.GetStringRef());
            GetGDBProcess().SetThreadStopInfo (stop_packet);
        }
    }
    return m_actual_stop_info_sp;
}


