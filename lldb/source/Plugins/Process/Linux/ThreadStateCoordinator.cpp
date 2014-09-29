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
#include <cstdarg>

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
        // Request a stop for all the thread stops that need to be stopped
        // and are not already known to be stopped.  Keep a list of all the
        // threads from which we still need to hear a stop reply.

        ThreadIDSet sent_tids;
        for (auto tid : m_wait_for_stop_tids)
        {
            // If we don't know about the thread's stop state or we
            // know it is not stopped, we need to send it a stop request.
            auto find_it = coordinator.m_tid_stop_map.find (tid);
            if ((find_it == coordinator.m_tid_stop_map.end ()) || !find_it->second)
            {
                m_request_thread_stop_func (tid);
                sent_tids.insert (tid);
            }
        }

        // We only need to wait for the sent_tids - so swap our wait set
        // to the sent tids.  The rest are already stopped and we won't
        // be receiving stop notifications for them.
        m_wait_for_stop_tids.swap (sent_tids);

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

    // Return true if still pending thread stops waiting; false if no more stops.
    // If no more pending stops, signal.
    bool
    RemoveThreadStopRequirementAndMaybeSignal (lldb::tid_t tid)
    {
        // Remove this tid if it was in it.
        m_wait_for_stop_tids.erase (tid);

        // Fire pending notification if no pending thread stops remain.
        if (m_wait_for_stop_tids.empty ())
        {
            // Fire the pending notification now.
            NotifyNow ();
            return false;
        }

        // Still have pending thread stops.
        return true;
    }

    void
    AddThreadStopRequirement (lldb::tid_t tid)
    {
        // Add this tid.
        auto insert_result = m_wait_for_stop_tids.insert (tid);

        // If it was really added, send the stop request to it.
        if (insert_result.second)
            m_request_thread_stop_func (tid);
    }

private:

    void
    NotifyNow ()
    {
        m_call_after_func (m_triggering_tid);
    }

    const lldb::tid_t m_triggering_tid;
    ThreadIDSet m_wait_for_stop_tids;
    ThreadIDFunc m_request_thread_stop_func;
    ThreadIDFunc m_call_after_func;
};

//===----------------------------------------------------------------------===//

class ThreadStateCoordinator::EventReset : public ThreadStateCoordinator::EventBase
{
public:
    EventReset ():
    EventBase ()
    {
    }

    bool
    ProcessEvent(ThreadStateCoordinator &coordinator) override
    {
        coordinator.ResetNow ();
        return true;
    }
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

class ThreadStateCoordinator::EventThreadCreate : public ThreadStateCoordinator::EventBase
{
public:
    EventThreadCreate (lldb::tid_t tid):
    EventBase (),
    m_tid (tid)
    {
    }

    bool
    ProcessEvent(ThreadStateCoordinator &coordinator) override
    {
        coordinator.ThreadWasCreated (m_tid);
        return true;
    }

private:

    const lldb::tid_t m_tid;
};

//===----------------------------------------------------------------------===//

class ThreadStateCoordinator::EventThreadDeath : public ThreadStateCoordinator::EventBase
{
public:
    EventThreadDeath (lldb::tid_t tid):
    EventBase (),
    m_tid (tid)
    {
    }

    bool
    ProcessEvent(ThreadStateCoordinator &coordinator) override
    {
        coordinator.ThreadDidDie (m_tid);
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
        EventCallAfterThreadsStop *const call_after_event = static_cast<EventCallAfterThreadsStop*> (m_pending_notification_sp.get ());
        const bool pending_stops_remain = call_after_event->RemoveThreadStopRequirementAndMaybeSignal (tid);
        if (!pending_stops_remain)
        {
            // Clear the pending notification now.
            m_pending_notification_sp.reset ();
        }
    }
}

void
ThreadStateCoordinator::ThreadWasCreated (lldb::tid_t tid)
{
    // Add the new thread to the stop map.
    // We assume a created thread is not stopped.
    m_tid_stop_map[tid] = false;

    if (m_pending_notification_sp)
    {
        // Tell the pending notification that we need to wait
        // for this new thread to stop.
        EventCallAfterThreadsStop *const call_after_event = static_cast<EventCallAfterThreadsStop*> (m_pending_notification_sp.get ());
        call_after_event->AddThreadStopRequirement (tid);
    }
}

void
ThreadStateCoordinator::ThreadDidDie (lldb::tid_t tid)
{
    // Update the global list of known thread states.  While this one is stopped, it is also dead.
    // So stop tracking it.  We assume the user of this coordinator will not keep trying to add
    // dependencies on a thread after it is known to be dead.
    m_tid_stop_map.erase (tid);

    // If we have a pending notification, remove this from the set.
    if (m_pending_notification_sp)
    {
        EventCallAfterThreadsStop *const call_after_event = static_cast<EventCallAfterThreadsStop*> (m_pending_notification_sp.get ());
        const bool pending_stops_remain = call_after_event->RemoveThreadStopRequirementAndMaybeSignal (tid);
        if (!pending_stops_remain)
        {
            // Clear the pending notification now.
            m_pending_notification_sp.reset ();
        }
    }
}

void
ThreadStateCoordinator::ResetNow ()
{
    // Clear the pending notification if there was one.
    m_pending_notification_sp.reset ();

    // Clear the stop map - we no longer know anything about any thread state.
    // The caller is expected to reset thread states for all threads, and we
    // will assume anything we haven't heard about is running and requires a
    // stop.
    m_tid_stop_map.clear ();
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
ThreadStateCoordinator::NotifyThreadCreate (lldb::tid_t tid)
{
    EnqueueEvent (EventBaseSP (new EventThreadCreate (tid)));
}

void
ThreadStateCoordinator::NotifyThreadDeath (lldb::tid_t tid)
{
    EnqueueEvent (EventBaseSP (new EventThreadDeath (tid)));
}

void
ThreadStateCoordinator::ResetForExec ()
{
    std::lock_guard<std::mutex> lock (m_queue_mutex);

    // Remove everything from the queue.  This is the only
    // state mutation that takes place outside the processing
    // loop.
    QueueType empty_queue;
    m_event_queue.swap (empty_queue);

    // Do the real clear behavior on the the queue to eliminate
    // the chance that processing of a dequeued earlier event is
    // overlapping with the clearing of state here.  Push it
    // directly because we need to have this happen with the lock,
    // and so far I only have this one place that needs a no-lock
    // variant.
    m_event_queue.push (EventBaseSP (new EventReset ()));
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
