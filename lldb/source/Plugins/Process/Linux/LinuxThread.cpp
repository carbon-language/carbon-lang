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
