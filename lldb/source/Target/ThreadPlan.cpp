//===-- ThreadPlan.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlan.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Thread.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/State.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ThreadPlan constructor
//----------------------------------------------------------------------
ThreadPlan::ThreadPlan(const char *name, Thread &thread, Vote stop_vote, Vote run_vote) :
    m_name (name),
    m_thread (thread),
    m_plan_complete(false),
    m_plan_complete_mutex (Mutex::eMutexTypeRecursive),
    m_plan_private (false),
    m_stop_vote (stop_vote),
    m_run_vote (run_vote),
    m_okay_to_discard (false)
{
    SetID (GetNextID());
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ThreadPlan::~ThreadPlan()
{
}

const char *
ThreadPlan::GetName () const
{
    return m_name.c_str();
}

Thread &
ThreadPlan::GetThread()
{
    return m_thread;
}


const Thread &
ThreadPlan::GetThread() const
{
    return m_thread;
}

bool
ThreadPlan::IsPlanComplete ()
{
    Mutex::Locker (m_plan_complete_mutex);
    return m_plan_complete;
}

void
ThreadPlan::SetPlanComplete ()
{
    Mutex::Locker (m_plan_complete_mutex);
    m_plan_complete = true;
}

bool
ThreadPlan::MischiefManaged ()
{
    Mutex::Locker (m_plan_complete_mutex);
    m_plan_complete = true;
    return true;
}

Vote
ThreadPlan::ShouldReportStop (Event *event_ptr)
{
    if (m_stop_vote == eVoteNoOpinion)
    {
        ThreadPlan *prev_plan = GetPreviousPlan ();
        if (prev_plan)
            return prev_plan->ShouldReportStop (event_ptr);
    }
    return m_stop_vote;
}

Vote
ThreadPlan::ShouldReportRun (Event *event_ptr)
{
    if (m_run_vote == eVoteNoOpinion)
    {
        ThreadPlan *prev_plan = GetPreviousPlan ();
        if (prev_plan)
            return prev_plan->ShouldReportRun (event_ptr);
    }
    return m_run_vote;
}

bool
ThreadPlan::StopOthers ()
{
    ThreadPlan *prev_plan;
    prev_plan = GetPreviousPlan ();
    if (prev_plan == NULL)
        return false;
    else
        return prev_plan->StopOthers();
}

bool
ThreadPlan::WillResume (StateType resume_state, bool current_plan)
{
    if (current_plan)
    {
        Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);

        if (log)
            log->Printf("Thread #%u: tid = 0x%4.4x about to resume the \"%s\" plan - state: %s - stop others: %d.", 
                        m_thread.GetIndexID(), m_thread.GetID(),  m_name.c_str(), StateAsCString(resume_state), StopOthers());
    }
    return true;
}

lldb::user_id_t
ThreadPlan::GetNextID()
{
    static uint32_t g_nextPlanID = 0;
    return ++g_nextPlanID;
}

void
ThreadPlan::DidPush()
{
}

void
ThreadPlan::WillPop()
{
}

void
ThreadPlan::PushPlan (ThreadPlanSP &thread_plan_sp)
{
    m_thread.PushPlan (thread_plan_sp);
}

ThreadPlan *
ThreadPlan::GetPreviousPlan ()
{
    return m_thread.GetPreviousPlan (this);
}

void
ThreadPlan::SetPrivate (bool input)
{
    m_plan_private = input;
}

bool
ThreadPlan::GetPrivate (void)
{
    return m_plan_private;
}

bool
ThreadPlan::OkayToDiscard()
{
    if (!IsMasterPlan())
        return true;
    else
        return m_okay_to_discard;
}

