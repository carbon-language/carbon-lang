//===-- LinuxThread.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "LinuxThread.h"
#include "lldb/Core/State.h"
#include "ProcessPOSIX.h"
#include "ProcessMonitor.h"
#include "ProcessPOSIXLog.h"

using namespace lldb;
using namespace lldb_private;

//------------------------------------------------------------------------------
// Constructors and destructors.

LinuxThread::LinuxThread(Process &process, lldb::tid_t tid)
    : POSIXThread(process, tid)
{
}

LinuxThread::~LinuxThread()
{
}

//------------------------------------------------------------------------------
// ProcessInterface protocol.

bool
LinuxThread::Resume()
{
    lldb::StateType resume_state = GetResumeState();
    ProcessMonitor &monitor = GetMonitor();
    bool status;

    Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_THREAD));
    if (log)
        log->Printf ("POSIXThread::%s (), resume_state = %s", __FUNCTION__,
                         StateAsCString(resume_state));

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
    case lldb::eStateStopped:
    case lldb::eStateSuspended:
        status = true;
        break;
    }

    return status;
}

void
LinuxThread::RefreshStateAfterStop()
{
    // Invalidate the thread names every time we get a stop event on Linux so we
    // will re-read the procfs comm virtual file when folks ask for the thread name.
    m_thread_name_valid = false;

    POSIXThread::RefreshStateAfterStop();
}

void
LinuxThread::TraceNotify(const ProcessMessage &message)
{
    POSIXBreakpointProtocol* reg_ctx = GetPOSIXBreakpointProtocol();
    if (reg_ctx)
    {
        uint32_t num_hw_wps = reg_ctx->NumSupportedHardwareWatchpoints();
        uint32_t wp_idx;
        for (wp_idx = 0; wp_idx < num_hw_wps; wp_idx++)
        {
            if (reg_ctx->IsWatchpointHit(wp_idx))
            {
                WatchNotify(message);
                return;
            }
        }
    }
    
    POSIXThread::TraceNotify (message);
}
