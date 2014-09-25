//===-- ThreadStateCoordinator.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ThreadStateCoordinator.h"

using namespace lldb_private;

ThreadStateCoordinator::ThreadStateCoordinator (const LogFunc &log_func) :
    m_done_b (false),
    m_log_func (log_func)
{
}

void
ThreadStateCoordinator::CallAfterThreadsStop (const lldb::tid_t triggering_tid,
                                              const ThreadIDSet &wait_for_stop_tids,
                                              const ThreadIDFunc &request_thread_stop_func,
                                              const ThreadIDFunc &call_after_func)
{
}

void
ThreadStateCoordinator::StopCoordinator ()
{
    m_done_b = true;
}

bool
ThreadStateCoordinator::ProcessNextEvent ()
{
    return !m_done_b;
}
