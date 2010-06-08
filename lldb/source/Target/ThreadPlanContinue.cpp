//===-- ThreadPlanContinue.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanContinue.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private-log.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ThreadPlanContinue: Continue plan
//----------------------------------------------------------------------

ThreadPlanContinue::ThreadPlanContinue (Thread &thread, bool stop_others, Vote stop_vote, Vote run_vote, bool immediate) :
    ThreadPlan ("Continue after previous plan", thread, stop_vote, run_vote),
    m_stop_others (stop_others),
    m_did_run (false),
    m_immediate (immediate)
{
}

ThreadPlanContinue::~ThreadPlanContinue ()
{
}

void
ThreadPlanContinue::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
        s->Printf ("continue");
    else
    {
        s->Printf ("Continue from the previous plan");
    }
}

bool
ThreadPlanContinue::ValidatePlan (Stream *error)
{
    // Since we read the instruction we're stepping over from the thread,
    // this plan will always work.
    return true;
}

bool
ThreadPlanContinue::PlanExplainsStop ()
{
    return true;
}

bool
ThreadPlanContinue::ShouldStop (Event *event_ptr)
{
    return false;
}

bool
ThreadPlanContinue::IsImmediate () const
{
    return m_immediate;
    return false;
}

bool
ThreadPlanContinue::StopOthers ()
{
    return m_stop_others;
}

StateType
ThreadPlanContinue::RunState ()
{
    return eStateRunning;
}

bool
ThreadPlanContinue::WillResume (StateType resume_state, bool current_plan)
{
    ThreadPlan::WillResume (resume_state, current_plan);
    if (current_plan)
    {
        m_did_run = true;
    }
    return true;
}

bool
ThreadPlanContinue::WillStop ()
{
    return true;
}

bool
ThreadPlanContinue::MischiefManaged ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);

    if (m_did_run)
    {
        if (log)
            log->Printf("Completed continue plan.");
        ThreadPlan::MischiefManaged ();
        return true;
    }
    else
        return false;
}
