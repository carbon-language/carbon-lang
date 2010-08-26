//===-- ThreadList.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdlib.h>

#include <algorithm>

#include "lldb/Target/ThreadList.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;

ThreadList::ThreadList (Process *process) :
    m_process (process),
    m_stop_id (0),
    m_threads(),
    m_threads_mutex (Mutex::eMutexTypeRecursive),
    m_selected_tid (LLDB_INVALID_THREAD_ID)
{
}

ThreadList::ThreadList (const ThreadList &rhs) :
    m_process (),
    m_stop_id (),
    m_threads (),
    m_threads_mutex (Mutex::eMutexTypeRecursive),
    m_selected_tid ()
{
    // Use the assignment operator since it uses the mutex
    *this = rhs;
}

const ThreadList&
ThreadList::operator = (const ThreadList& rhs)
{
    if (this != &rhs)
    {
        // Lock both mutexes to make sure neither side changes anyone on us
        // while the assignement occurs
        Mutex::Locker locker_lhs(m_threads_mutex);
        Mutex::Locker locker_rhs(rhs.m_threads_mutex);
        m_process = rhs.m_process;
        m_stop_id = rhs.m_stop_id;
        m_threads = rhs.m_threads;
        m_selected_tid = rhs.m_selected_tid;
    }
    return *this;
}


ThreadList::~ThreadList()
{
}


uint32_t
ThreadList::GetStopID () const
{
    return m_stop_id;
}

void
ThreadList::SetStopID (uint32_t stop_id)
{
    m_stop_id = stop_id;
}


void
ThreadList::AddThread (ThreadSP &thread_sp)
{
    Mutex::Locker locker(m_threads_mutex);
    m_threads.push_back(thread_sp);
}

uint32_t
ThreadList::GetSize (bool can_update)
{
    Mutex::Locker locker(m_threads_mutex);
    if (can_update)
        m_process->UpdateThreadListIfNeeded();
    return m_threads.size();
}

ThreadSP
ThreadList::GetThreadAtIndex (uint32_t idx, bool can_update)
{
    Mutex::Locker locker(m_threads_mutex);
    if (can_update)
        m_process->UpdateThreadListIfNeeded();

    ThreadSP thread_sp;
    if (idx < m_threads.size())
        thread_sp = m_threads[idx];
    return thread_sp;
}

ThreadSP
ThreadList::FindThreadByID (lldb::tid_t tid, bool can_update)
{
    Mutex::Locker locker(m_threads_mutex);

    if (can_update)
        m_process->UpdateThreadListIfNeeded();

    ThreadSP thread_sp;
    uint32_t idx = 0;
    const uint32_t num_threads = m_threads.size();
    for (idx = 0; idx < num_threads; ++idx)
    {
        if (m_threads[idx]->GetID() == tid)
        {
            thread_sp = m_threads[idx];
            break;
        }
    }
    return thread_sp;
}

ThreadSP
ThreadList::GetThreadSPForThreadPtr (Thread *thread_ptr)
{
    ThreadSP thread_sp;
    if (thread_ptr)
    {
        Mutex::Locker locker(m_threads_mutex);

        uint32_t idx = 0;
        const uint32_t num_threads = m_threads.size();
        for (idx = 0; idx < num_threads; ++idx)
        {
            if (m_threads[idx].get() == thread_ptr)
            {
                thread_sp = m_threads[idx];
                break;
            }
        }
    }
    return thread_sp;
}



ThreadSP
ThreadList::FindThreadByIndexID (uint32_t index_id, bool can_update)
{
    Mutex::Locker locker(m_threads_mutex);

    if (can_update)
        m_process->UpdateThreadListIfNeeded();

    ThreadSP thread_sp;
    const uint32_t num_threads = m_threads.size();
    for (uint32_t idx = 0; idx < num_threads; ++idx)
    {
        if (m_threads[idx]->GetIndexID() == index_id)
        {
            thread_sp = m_threads[idx];
            break;
        }
    }
    return thread_sp;
}

bool
ThreadList::ShouldStop (Event *event_ptr)
{
    Mutex::Locker locker(m_threads_mutex);

    // Running events should never stop, obviously...


    bool should_stop = false;    
    m_process->UpdateThreadListIfNeeded();

    collection::iterator pos, end = m_threads.end();

    // Run through the threads and ask whether we should stop.  Don't ask
    // suspended threads, however, it makes more sense for them to preserve their
    // state across the times the process runs but they don't get a chance to.
    for (pos = m_threads.begin(); pos != end; ++pos)
    {
        ThreadSP thread_sp(*pos);
        if ((thread_sp->GetResumeState () != eStateSuspended) && (thread_sp->ThreadStoppedForAReason()))
        {
            should_stop |=  thread_sp->ShouldStop(event_ptr);
        }
    }

    if (should_stop)
    {
        for (pos = m_threads.begin(); pos != end; ++pos)
        {
            ThreadSP thread_sp(*pos);
            thread_sp->WillStop ();
        }
    }

    return should_stop;
}

Vote
ThreadList::ShouldReportStop (Event *event_ptr)
{
    Vote result = eVoteNoOpinion;
    m_process->UpdateThreadListIfNeeded();
    collection::iterator pos, end = m_threads.end();

    // Run through the threads and ask whether we should report this event.
    // For stopping, a YES vote wins over everything.  A NO vote wins over NO opinion.
    for (pos = m_threads.begin(); pos != end; ++pos)
    {
        ThreadSP thread_sp(*pos);
        if (thread_sp->ThreadStoppedForAReason() && (thread_sp->GetResumeState () != eStateSuspended))
        {
            switch (thread_sp->ShouldReportStop (event_ptr))
            {
                case eVoteNoOpinion:
                    continue;
                case eVoteYes:
                    result = eVoteYes;
                    break;
                case eVoteNo:
                    if (result == eVoteNoOpinion)
                        result = eVoteNo;
                    break;
            }
        }
    }
    return result;
}

Vote
ThreadList::ShouldReportRun (Event *event_ptr)
{
    Vote result = eVoteNoOpinion;
    m_process->UpdateThreadListIfNeeded();
    collection::iterator pos, end = m_threads.end();

    // Run through the threads and ask whether we should report this event.
    // The rule is NO vote wins over everything, a YES vote wins over no opinion.

    for (pos = m_threads.begin(); pos != end; ++pos)
    {
        ThreadSP thread_sp(*pos);
        if (thread_sp->GetResumeState () != eStateSuspended)

        switch (thread_sp->ShouldReportRun (event_ptr))
        {
            case eVoteNoOpinion:
                continue;
            case eVoteYes:
                if (result == eVoteNoOpinion)
                    result = eVoteYes;
                break;
            case eVoteNo:
                result = eVoteNo;
                break;
        }
    }
    return result;
}

void
ThreadList::Clear()
{
    m_stop_id = 0;
    m_threads.clear();
    m_selected_tid = LLDB_INVALID_THREAD_ID;
}

void
ThreadList::RefreshStateAfterStop ()
{
    Mutex::Locker locker(m_threads_mutex);

    m_process->UpdateThreadListIfNeeded();

    collection::iterator pos, end = m_threads.end();
    for (pos = m_threads.begin(); pos != end; ++pos)
        (*pos)->RefreshStateAfterStop ();
}

void
ThreadList::DiscardThreadPlans ()
{
    // You don't need to update the thread list here, because only threads
    // that you currently know about have any thread plans.
    Mutex::Locker locker(m_threads_mutex);

    collection::iterator pos, end = m_threads.end();
    for (pos = m_threads.begin(); pos != end; ++pos)
        (*pos)->DiscardThreadPlans (true);

}

bool
ThreadList::WillResume ()
{
    // Run through the threads and perform their momentary actions.
    // But we only do this for threads that are running, user suspended
    // threads stay where they are.
    bool success = true;

    Mutex::Locker locker(m_threads_mutex);
    m_process->UpdateThreadListIfNeeded();

    collection::iterator pos, end = m_threads.end();

    // See if any thread wants to run stopping others.  If it does, then we won't
    // setup the other threads for resume, since they aren't going to get a chance
    // to run.  This is necessary because the SetupForResume might add "StopOthers"
    // plans which would then get to be part of the who-gets-to-run negotiation, but
    // they're coming in after the fact, and the threads that are already set up should
    // take priority.
    
    bool wants_solo_run = false;
    
    for (pos = m_threads.begin(); pos != end; ++pos)
    {
        if ((*pos)->GetResumeState() != eStateSuspended &&
                 (*pos)->GetCurrentPlan()->StopOthers())
        {
            wants_solo_run = true;
            break;
        }
    }   

    
    // Give all the threads that are likely to run a last chance to set up their state before we
    // negotiate who is actually going to get a chance to run...
    // Don't set to resume suspended threads, and if any thread wanted to stop others, only
    // call setup on the threads that request StopOthers...
    
    for (pos = m_threads.begin(); pos != end; ++pos)
    {
        if ((*pos)->GetResumeState() != eStateSuspended
            && (!wants_solo_run || (*pos)->GetCurrentPlan()->StopOthers()))
        {
            (*pos)->SetupForResume ();
        }
    }

    // Now go through the threads and see if any thread wants to run just itself.
    // if so then pick one and run it.
    
    ThreadList run_me_only_list (m_process);
    
    run_me_only_list.SetStopID(m_process->GetStopID());

    ThreadSP immediate_thread_sp;
    bool run_only_current_thread = false;

    for (pos = m_threads.begin(); pos != end; ++pos)
    {
        ThreadSP thread_sp(*pos);
        if (thread_sp->GetCurrentPlan()->IsImmediate())
        {
            // We first do all the immediate plans, so if we find one, set
            // immediate_thread_sp and break out, and we'll pick it up first thing
            // when we're negotiating which threads get to run.
            immediate_thread_sp = thread_sp;
            break;
        }
        else if (thread_sp->GetResumeState() != eStateSuspended &&
                 thread_sp->GetCurrentPlan()->StopOthers())
        {
            // You can't say "stop others" and also want yourself to be suspended.
            assert (thread_sp->GetCurrentPlan()->RunState() != eStateSuspended);

            if (thread_sp == GetSelectedThread())
            {
                run_only_current_thread = true;
                run_me_only_list.Clear();
                run_me_only_list.AddThread (thread_sp);
                break;
            }

            run_me_only_list.AddThread (thread_sp);
        }

    }

    if (immediate_thread_sp)
    {
        for (pos = m_threads.begin(); pos != end; ++pos)
        {
            ThreadSP thread_sp(*pos);
            if (thread_sp.get() == immediate_thread_sp.get())
                thread_sp->WillResume(thread_sp->GetCurrentPlan()->RunState());
            else
                thread_sp->WillResume (eStateSuspended);
        }
    }
    else if (run_me_only_list.GetSize (false) == 0)
    {
        // Everybody runs as they wish:
        for (pos = m_threads.begin(); pos != end; ++pos)
        {
            ThreadSP thread_sp(*pos);
            thread_sp->WillResume(thread_sp->GetCurrentPlan()->RunState());
        }
    }
    else
    {
        ThreadSP thread_to_run;

        if (run_only_current_thread)
        {
            thread_to_run = GetSelectedThread();
        }
        else if (run_me_only_list.GetSize (false) == 1)
        {
            thread_to_run = run_me_only_list.GetThreadAtIndex (0);
        }
        else
        {
            int random_thread = (int)
                    ((run_me_only_list.GetSize (false) * (double) rand ()) / (RAND_MAX + 1.0));
            thread_to_run = run_me_only_list.GetThreadAtIndex (random_thread);
        }

        for (pos = m_threads.begin(); pos != end; ++pos)
        {
            ThreadSP thread_sp(*pos);
            if (thread_sp == thread_to_run)
                thread_sp->WillResume(thread_sp->GetCurrentPlan()->RunState());
            else
                thread_sp->WillResume (eStateSuspended);
        }
    }

    return success;
}

void
ThreadList::DidResume ()
{
    collection::iterator pos, end = m_threads.end();
    for (pos = m_threads.begin(); pos != end; ++pos)
    {
        // Don't clear out threads that aren't going to get a chance to run, rather
        // leave their state for the next time around.
        ThreadSP thread_sp(*pos);
        if (thread_sp->GetResumeState() != eStateSuspended)
            thread_sp->DidResume ();
    }
}

ThreadSP
ThreadList::GetSelectedThread ()
{
    Mutex::Locker locker(m_threads_mutex);
    return FindThreadByID(m_selected_tid);
}

bool
ThreadList::SetSelectedThreadByID (lldb::tid_t tid)
{
    Mutex::Locker locker(m_threads_mutex);
    if  (FindThreadByID(tid).get())
        m_selected_tid = tid;
    else
        m_selected_tid = LLDB_INVALID_THREAD_ID;

    return m_selected_tid != LLDB_INVALID_THREAD_ID;
}

bool
ThreadList::SetSelectedThreadByIndexID (uint32_t index_id)
{
    Mutex::Locker locker(m_threads_mutex);
    ThreadSP thread_sp (FindThreadByIndexID(index_id));
    if  (thread_sp.get())
        m_selected_tid = thread_sp->GetID();
    else
        m_selected_tid = LLDB_INVALID_THREAD_ID;

    return m_selected_tid != LLDB_INVALID_THREAD_ID;
}

