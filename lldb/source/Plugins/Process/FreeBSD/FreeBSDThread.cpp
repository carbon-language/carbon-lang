//===-- FreeBSDThread.cpp ---------------------------------------*- C++ -*-===//
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
#include "lldb/Core/State.h"

// Project includes
#include "FreeBSDThread.h"
#include "ProcessFreeBSD.h"
#include "ProcessPOSIXLog.h"

using namespace lldb;
using namespace lldb_private;

//------------------------------------------------------------------------------
// Constructors and destructors.

FreeBSDThread::FreeBSDThread(Process &process, lldb::tid_t tid)
    : POSIXThread(process, tid)
{
}

FreeBSDThread::~FreeBSDThread()
{
}

//------------------------------------------------------------------------------
// ProcessInterface protocol.

void
FreeBSDThread::WillResume(lldb::StateType resume_state)
{
    Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_THREAD));
    if (log)
        log->Printf("tid %lu resume_state = %s", GetID(),
                    lldb_private::StateAsCString(resume_state));
    ProcessSP process_sp(GetProcess());
    ProcessFreeBSD *process = static_cast<ProcessFreeBSD *>(process_sp.get());
    int signo = GetResumeSignal();
    bool signo_valid = process->GetUnixSignals().SignalIsValid(signo);

    switch (resume_state)
    {
    case eStateSuspended:
    case eStateStopped:
        process->m_suspend_tids.push_back(GetID());
        break;
    case eStateRunning:
        process->m_run_tids.push_back(GetID());
        if (signo_valid)
            process->m_resume_signo = signo;
        break;
    case eStateStepping:
        process->m_step_tids.push_back(GetID());
        if (signo_valid)
            process->m_resume_signo = signo;
        break; 
    default:
        break;
    }
}
