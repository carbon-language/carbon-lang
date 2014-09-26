//===-- ThreadStateCoordinator.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#if !defined (__STDC_FORMAT_MACROS)
#define __STDC_FORMAT_MACROS 1
#endif

#include <inttypes.h>

#include "ThreadStateCoordinator.h"
#include <memory>

using namespace lldb_private;

//===----------------------------------------------------------------------===//

class ThreadStateCoordinator::EventBase : public std::enable_shared_from_this<ThreadStateCoordinator::EventBase>
{
public:
    EventBase ()
    {
    }

    virtual
    ~EventBase ()
    {
    }

    // Return false if the coordinator should terminate running.
    virtual bool
    ProcessEvent (ThreadStateCoordinator &coordinator) = 0;
};

//===----------------------------------------------------------------------===//

class ThreadStateCoordinator::EventStopCoordinator : public ThreadStateCoordinator::EventBase
{
public:
    EventStopCoordinator ():
        EventBase ()
    {
    }

    ~EventStopCoordinator () override
    {
    }

    bool
    ProcessEvent(ThreadStateCoordinator &coordinator) override
    {
        return false;
    }
};

//===----------------------------------------------------------------------===//

class ThreadStateCoordinator::EventCallAfterThreadsStop : public ThreadStateCoordinator::EventBase
{
public:
    EventCallAfterThreadsStop (lldb::tid_t triggering_tid,
                               const ThreadIDSet &wait_for_stop_tids,
                               const ThreadIDFunc &request_thread_stop_func,
                               const ThreadIDFunc &call_after_func):
    EventBase (),
    m_triggering_tid (triggering_tid),
    m_wait_for_stop_tids (wait_for_stop_tids),
    m_request_thread_stop_func (request_thread_stop_func),
    m_call_after_func (call_after_func)
    {
    }

    ~EventCallAfterThreadsStop () override
    {
    }

    lldb::tid_t GetTriggeringTID () const
    {
        return m_triggering_tid;
    }

    ThreadIDSet &
    GetRemainingWaitTIDs ()
    {
        return m_wait_for_stop_tids;
    }

    bool
    ProcessEvent(ThreadStateCoordinator &coordinator) override
    {
        // Request a stop for all the thread stops that are needed.
        // In the future, we can keep track of which stops we're
        // still waiting for, and can not re-issue for already
        // requested stops.
        for (auto tid : m_wait_for_stop_tids)
            m_request_thread_stop_func (tid);

        if (m_wait_for_stop_tids.empty ())
        {
        // We're not waiting for any threads.  Fire off the deferred signal delivery event.
            NotifyNow ();
        }
        else
        {
            // The real meat of this class: wait until all
            // the thread stops (or thread deaths) come in
            // before firing off that the triggering signal
            // arrived.
            coordinator.SetPendingNotification (shared_from_this ());
        }

        return true;
    }

    void
    NotifyNow ()
    {
        m_call_after_func (m_triggering_tid);
    }

private:

    const lldb::tid_t m_triggering_tid;
    ThreadIDSet m_wait_for_stop_tids;
    ThreadIDFunc m_request_thread_stop_func;
    ThreadIDFunc m_call_after_func;
};

//===----------------------------------------------------------------------===//

class ThreadStateCoordinator::EventThreadStopped : public ThreadStateCoordinator::EventBase
{
public:
    EventThreadStopped (lldb::tid_t tid):
    EventBase (),
    m_tid (tid)
    {
    }

    ~EventThreadStopped () override
    {
    }

    bool
    ProcessEvent(ThreadStateCoordinator &coordinator) override
    {
        coordinator.ThreadDidStop (m_tid);
        return true;
    }

private:

    const lldb::tid_t m_tid;
};

//===----------------------------------------------------------------------===//

ThreadStateCoordinator::ThreadStateCoordinator (const LogFunc &log_func) :
    m_log_func (log_func),
    m_event_queue (),
    m_queue_condition (),
    m_queue_mutex (),
    m_tid_stop_map ()
{
}

void
ThreadStateCoordinator::EnqueueEvent (EventBaseSP event_sp)
{
    std::lock_guard<std::mutex> lock (m_queue_mutex);
    m_event_queue.push (event_sp);
    m_queue_condition.notify_one ();
}

ThreadStateCoordinator::EventBaseSP
ThreadStateCoordinator::DequeueEventWithWait ()
{
    // Wait for an event to be present.
    std::unique_lock<std::mutex> lock (m_queue_mutex);
    m_queue_condition.wait (lock,
                            [this] { return !m_event_queue.empty (); });

    // Grab the event and pop it off the queue.
    EventBaseSP event_sp = m_event_queue.front ();
    m_event_queue.pop ();

    return event_sp;
}

void
ThreadStateCoordinator::SetPendingNotification (const EventBaseSP &event_sp)
{
    assert (event_sp && "null event_sp");
    if (!event_sp)
        return;

    const EventCallAfterThreadsStop *new_call_after_event = static_cast<EventCallAfterThreadsStop*> (event_sp.get ());

    if (m_pending_notification_sp)
    {
        const EventCallAfterThreadsStop *prev_call_after_event = static_cast<EventCallAfterThreadsStop*> (m_pending_notification_sp.get ());

        // Yikes - we've already got a pending signal notification in progress.
        // Log this info.  We lose the pending notification here.
        Log ("ThreadStateCoordinator::%s dropping existing pending signal notification for tid %" PRIu64 ", to be replaced with signal for tid %" PRIu64,
                   __FUNCTION__,
                   prev_call_after_event->GetTriggeringTID (),
                   new_call_after_event->GetTriggeringTID ());
    }

    m_pending_notification_sp = event_sp;
}

void
ThreadStateCoordinator::CallAfterThreadsStop (const lldb::tid_t triggering_tid,
                                              const ThreadIDSet &wait_for_stop_tids,
                                              const ThreadIDFunc &request_thread_stop_func,
                                              const ThreadIDFunc &call_after_func)
{
    EnqueueEvent (EventBaseSP (new EventCallAfterThreadsStop (triggering_tid,
                                                              wait_for_stop_tids,
                                                              request_thread_stop_func,
                                                              call_after_func)));
}

void
ThreadStateCoordinator::ThreadDidStop (lldb::tid_t tid)
{
    // Update the global list of known thread states.  This one is definitely stopped.
    m_tid_stop_map[tid] = true;

    // If we have a pending notification, remove this from the set.
    if (m_pending_notification_sp)
    {
        EventCallAfterThreadsStop *call_after_event = static_cast<EventCallAfterThreadsStop*> (m_pending_notification_sp.get ());

        auto remaining_stop_tids = call_after_event->GetRemainingWaitTIDs ();

        // Remove this tid if it was in it.
        remaining_stop_tids.erase (tid);
        if (remaining_stop_tids.empty ())
        {
            // Fire the pending notification now.
            call_after_event->NotifyNow ();

            // Clear the pending notification now.
            m_pending_notification_sp.reset ();
        }
    }
}

void
ThreadStateCoordinator::Log (const char *format, ...)
{
    va_list args;
    va_start (args, format);

    m_log_func (format, args);

    va_end (args);
}

void
ThreadStateCoordinator::NotifyThreadStop (lldb::tid_t tid)
{
    EnqueueEvent (EventBaseSP (new EventThreadStopped (tid)));
}

void
ThreadStateCoordinator::StopCoordinator ()
{
    EnqueueEvent (EventBaseSP (new EventStopCoordinator ()));
}

bool
ThreadStateCoordinator::ProcessNextEvent ()
{
    return DequeueEventWithWait()->ProcessEvent (*this);
}
