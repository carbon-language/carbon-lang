//===-- ThreadKDP.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "ThreadKDP.h"

#include "llvm/Support/MachO.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/State.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Unwind.h"
#include "lldb/Breakpoint/WatchpointLocation.h"

#include "ProcessKDP.h"
#include "ProcessKDPLog.h"
#include "RegisterContextKDP_arm.h"
#include "RegisterContextKDP_i386.h"
#include "RegisterContextKDP_x86_64.h"
#include "Plugins/Process/Utility/UnwindLLDB.h"

#if defined(__APPLE__)
#include "UnwindMacOSXFrameBackchain.h"
#endif

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Thread Registers
//----------------------------------------------------------------------

ThreadKDP::ThreadKDP (ProcessKDP &process, lldb::tid_t tid) :
    Thread(process, tid),
    m_thread_name (),
    m_dispatch_queue_name (),
    m_thread_dispatch_qaddr (LLDB_INVALID_ADDRESS)
{
    ProcessKDPLog::LogIf(KDP_LOG_THREAD, "%p: ThreadKDP::ThreadKDP (pid = %i, tid = 0x%4.4x)", this, m_process.GetID(), GetID());
}

ThreadKDP::~ThreadKDP ()
{
    ProcessKDPLog::LogIf(KDP_LOG_THREAD, "%p: ThreadKDP::~ThreadKDP (pid = %i, tid = 0x%4.4x)", this, m_process.GetID(), GetID());
    DestroyThread();
}


const char *
ThreadKDP::GetInfo ()
{
    return NULL;
}


const char *
ThreadKDP::GetName ()
{
    if (m_thread_name.empty())
        return NULL;
    return m_thread_name.c_str();
}

const char *
ThreadKDP::GetQueueName ()
{
    return NULL;
}

bool
ThreadKDP::WillResume (StateType resume_state)
{
    ClearStackFrames();
    // Call the Thread::WillResume first. If we stop at a signal, the stop info
    // class for signal will set the resume signal that we need below. The signal
    // stuff obeys the Process::UnixSignal defaults. 
    Thread::WillResume(resume_state);

    lldb::LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
        log->Printf ("Resuming thread: %4.4x with state: %s.", GetID(), StateAsCString(resume_state));

//    ProcessKDP &process = GetKDPProcess();
//    switch (resume_state)
//    {
//    case eStateSuspended:
//    case eStateStopped:
//        // Don't append anything for threads that should stay stopped.
//        break;
//
//    case eStateRunning:
//    case eStateStepping:
//        break;
//
//    default:
//        break;
//    }
    return true;
}

void
ThreadKDP::RefreshStateAfterStop()
{
    // Invalidate all registers in our register context. We don't set "force" to
    // true because the stop reply packet might have had some register values
    // that were expedited and these will already be copied into the register
    // context by the time this function gets called. The KDPRegisterContext
    // class has been made smart enough to detect when it needs to invalidate
    // which registers are valid by putting hooks in the register read and 
    // register supply functions where they check the process stop ID and do
    // the right thing.
    const bool force = false;
    GetRegisterContext()->InvalidateIfNeeded (force);
}

Unwind *
ThreadKDP::GetUnwinder ()
{
    if (m_unwinder_ap.get() == NULL)
    {
        const ArchSpec target_arch (GetProcess().GetTarget().GetArchitecture ());
        const llvm::Triple::ArchType machine = target_arch.GetMachine();
        switch (machine)
        {
            case llvm::Triple::x86_64:
            case llvm::Triple::x86:
            case llvm::Triple::arm:
            case llvm::Triple::thumb:
                m_unwinder_ap.reset (new UnwindLLDB (*this));
                break;

            default:
#if defined(__APPLE__)
                m_unwinder_ap.reset (new UnwindMacOSXFrameBackchain (*this));
#endif
                break;
        }
    }
    return m_unwinder_ap.get();
}

void
ThreadKDP::ClearStackFrames ()
{
    Unwind *unwinder = GetUnwinder ();
    if (unwinder)
        unwinder->Clear();
    Thread::ClearStackFrames();
}


bool
ThreadKDP::ThreadIDIsValid (lldb::tid_t thread)
{
    return thread != 0;
}

void
ThreadKDP::Dump(Log *log, uint32_t index)
{
}


bool
ThreadKDP::ShouldStop (bool &step_more)
{
    return true;
}
lldb::RegisterContextSP
ThreadKDP::GetRegisterContext ()
{
    if (m_reg_context_sp.get() == NULL)
        m_reg_context_sp = CreateRegisterContextForFrame (NULL);
    return m_reg_context_sp;
}

lldb::RegisterContextSP
ThreadKDP::CreateRegisterContextForFrame (StackFrame *frame)
{
    lldb::RegisterContextSP reg_ctx_sp;
    uint32_t concrete_frame_idx = 0;
    
    if (frame)
        concrete_frame_idx = frame->GetConcreteFrameIndex ();

    if (concrete_frame_idx == 0)
    {
        switch (GetKDPProcess().GetCommunication().GetCPUType())
        {
            case llvm::MachO::CPUTypeARM:
                reg_ctx_sp.reset (new RegisterContextKDP_arm (*this, concrete_frame_idx));
                break;
            case llvm::MachO::CPUTypeI386:
                reg_ctx_sp.reset (new RegisterContextKDP_i386 (*this, concrete_frame_idx));
                break;
            case llvm::MachO::CPUTypeX86_64:
                reg_ctx_sp.reset (new RegisterContextKDP_x86_64 (*this, concrete_frame_idx));
                break;
            default:
                assert (!"Add CPU type support in KDP");
                break;
        }
    }
    else if (m_unwinder_ap.get())
        reg_ctx_sp = m_unwinder_ap->CreateRegisterContextForFrame (frame);
    return reg_ctx_sp;
}

lldb::StopInfoSP
ThreadKDP::GetPrivateStopReason ()
{
    const uint32_t process_stop_id = GetProcess().GetStopID();
    if (m_thread_stop_reason_stop_id != process_stop_id ||
        (m_actual_stop_info_sp && !m_actual_stop_info_sp->IsValid()))
    {
        // TODO: can we query the initial state of the thread here?
        // For now I am just going to pretend that a SIGSTOP happened.

        SetStopInfo(StopInfo::CreateStopReasonWithSignal (*this, SIGSTOP));

        // If GetKDPProcess().SetThreadStopInfo() doesn't find a stop reason
        // for this thread, then m_actual_stop_info_sp will not ever contain
        // a valid stop reason and the "m_actual_stop_info_sp->IsValid() == false"
        // check will never be able to tell us if we have the correct stop info
        // for this thread and we will continually send qThreadStopInfo packets
        // down to the remote KDP server, so we need to keep our own notion
        // of the stop ID that m_actual_stop_info_sp is valid for (even if it
        // contains nothing). We use m_thread_stop_reason_stop_id for this below.
//        m_thread_stop_reason_stop_id = process_stop_id;
//        m_actual_stop_info_sp.reset();

    }
    return m_actual_stop_info_sp;
}


