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
#include <sstream>

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
    virtual EventLoopResult
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

    EventLoopResult
    ProcessEvent(ThreadStateCoordinator &coordinator) override
    {
        return eventLoopResultStop;
    }
};

//===----------------------------------------------------------------------===//

class ThreadStateCoordinator::EventCallAfterThreadsStop : public ThreadStateCoordinator::EventBase
{
public:
    EventCallAfterThreadsStop (lldb::tid_t triggering_tid,
                               const ThreadIDSet &wait_for_stop_tids,
                               const ThreadIDFunction &request_thread_stop_function,
                               const ThreadIDFunction &call_after_function,
                               const ErrorFunction &error_function):
    EventBase (),
    m_triggering_tid (triggering_tid),
    m_wait_for_stop_tids (wait_for_stop_tids),
    m_original_wait_for_stop_tids (wait_for_stop_tids),
    m_request_thread_stop_function (request_thread_stop_function),
    m_call_after_function (call_after_function),
    m_error_function (error_function),
    m_request_stop_on_all_unstopped_threads (false)
    {
    }

    EventCallAfterThreadsStop (lldb::tid_t triggering_tid,
                               const ThreadIDFunction &request_thread_stop_function,
                               const ThreadIDFunction &call_after_function,
                               const ErrorFunction &error_function) :
    EventBase (),
    m_triggering_tid (triggering_tid),
    m_wait_for_stop_tids (),
    m_original_wait_for_stop_tids (),
    m_request_thread_stop_function (request_thread_stop_function),
    m_call_after_function (call_after_function),
    m_error_function (error_function),
    m_request_stop_on_all_unstopped_threads (true)
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

    const ThreadIDSet &
    GetRemainingWaitTIDs () const
    {
        return m_wait_for_stop_tids;
    }


    const ThreadIDSet &
    GetInitialWaitTIDs () const
    {
        return m_original_wait_for_stop_tids;
    }

    EventLoopResult
    ProcessEvent(ThreadStateCoordinator &coordinator) override
    {
        // Validate we know about the deferred trigger thread.
        if (coordinator.m_tid_stop_map.find (m_triggering_tid) == coordinator.m_tid_stop_map.end ())
        {
            // We don't know about this thread.  This is an error condition.
            std::ostringstream error_message;
            error_message << "error: deferred notification tid " << m_triggering_tid << " is unknown";
            m_error_function (error_message.str ());

            // We bail out here.
            return eventLoopResultContinue;
        }

        if (m_request_stop_on_all_unstopped_threads)
        {
            RequestStopOnAllRunningThreads (coordinator);
        }
        else
        {
            if (!RequestStopOnAllSpecifiedThreads (coordinator))
                return eventLoopResultContinue;
        }

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

        return eventLoopResultContinue;
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
            m_request_thread_stop_function (tid);
    }

private:

    void
    NotifyNow ()
    {
        m_call_after_function (m_triggering_tid);
    }

    bool
    RequestStopOnAllSpecifiedThreads (const ThreadStateCoordinator &coordinator)
    {
        // Request a stop for all the thread stops that need to be stopped
        // and are not already known to be stopped.  Keep a list of all the
        // threads from which we still need to hear a stop reply.

        ThreadIDSet sent_tids;
        for (auto tid : m_wait_for_stop_tids)
        {
            // Validate we know about all tids for which we must first receive a stop before
            // triggering the deferred stop notification.
            auto find_it = coordinator.m_tid_stop_map.find (tid);
            if (find_it == coordinator.m_tid_stop_map.end ())
            {
                // This is an error.  We shouldn't be asking for waiting pids that aren't known.
                // NOTE: we may be stripping out the specification of wait tids and handle this
                // automatically, in which case this state can never occur.
                std::ostringstream error_message;
                error_message << "error: deferred notification for tid " << m_triggering_tid << " specified an unknown/untracked pending stop tid " << m_triggering_tid;
                m_error_function (error_message.str ());

                // Bail out here.
                return false;
            }

            // If the pending stop thread is currently running, we need to send it a stop request.
            if (!find_it->second)
            {
                m_request_thread_stop_function (tid);
                sent_tids.insert (tid);
            }
        }

        // We only need to wait for the sent_tids - so swap our wait set
        // to the sent tids.  The rest are already stopped and we won't
        // be receiving stop notifications for them.
        m_wait_for_stop_tids.swap (sent_tids);

        // Succeeded, keep running.
        return true;
    }

    void
    RequestStopOnAllRunningThreads (const ThreadStateCoordinator &coordinator)
    {
        // Request a stop for all the thread stops that need to be stopped
        // and are not already known to be stopped.  Keep a list of all the
        // threads from which we still need to hear a stop reply.

        ThreadIDSet sent_tids;
        for (auto it = coordinator.m_tid_stop_map.begin(); it != coordinator.m_tid_stop_map.end(); ++it)
        {
            // We only care about threads not stopped.
            const bool running = !it->second;
            if (running)
            {
                // Request this thread stop.
                const lldb::tid_t tid = it->first;
                m_request_thread_stop_function (tid);
                sent_tids.insert (tid);
            }
        }

        // Set the wait list to the set of tids for which we requested stops.
        m_wait_for_stop_tids.swap (sent_tids);
    }

    const lldb::tid_t m_triggering_tid;
    ThreadIDSet m_wait_for_stop_tids;
    const ThreadIDSet m_original_wait_for_stop_tids;
    ThreadIDFunction m_request_thread_stop_function;
    ThreadIDFunction m_call_after_function;
    ErrorFunction m_error_function;
    const bool m_request_stop_on_all_unstopped_threads;
};

//===----------------------------------------------------------------------===//

class ThreadStateCoordinator::EventReset : public ThreadStateCoordinator::EventBase
{
public:
    EventReset ():
    EventBase ()
    {
    }

    EventLoopResult
    ProcessEvent(ThreadStateCoordinator &coordinator) override
    {
        coordinator.ResetNow ();
        return eventLoopResultContinue;
    }
};

//===----------------------------------------------------------------------===//

class ThreadStateCoordinator::EventThreadStopped : public ThreadStateCoordinator::EventBase
{
public:
    EventThreadStopped (lldb::tid_t tid,
                        const ErrorFunction &error_function):
    EventBase (),
    m_tid (tid),
    m_error_function (error_function)
    {
    }

    EventLoopResult
    ProcessEvent(ThreadStateCoordinator &coordinator) override
    {
        coordinator.ThreadDidStop (m_tid, m_error_function);
        return eventLoopResultContinue;
    }

private:

    const lldb::tid_t m_tid;
    ErrorFunction m_error_function;
};

//===----------------------------------------------------------------------===//

class ThreadStateCoordinator::EventThreadCreate : public ThreadStateCoordinator::EventBase
{
public:
    EventThreadCreate (lldb::tid_t tid,
                       bool is_stopped,
                       const ErrorFunction &error_function):
    EventBase (),
    m_tid (tid),
    m_is_stopped (is_stopped),
    m_error_function (error_function)
    {
    }

    EventLoopResult
    ProcessEvent(ThreadStateCoordinator &coordinator) override
    {
        coordinator.ThreadWasCreated (m_tid, m_is_stopped, m_error_function);
        return eventLoopResultContinue;
    }

private:

    const lldb::tid_t m_tid;
    const bool m_is_stopped;
    ErrorFunction m_error_function;
};

//===----------------------------------------------------------------------===//

class ThreadStateCoordinator::EventThreadDeath : public ThreadStateCoordinator::EventBase
{
public:
    EventThreadDeath (lldb::tid_t tid,
                      const ErrorFunction &error_function):
    EventBase (),
    m_tid (tid),
    m_error_function (error_function)
    {
    }

    EventLoopResult
    ProcessEvent(ThreadStateCoordinator &coordinator) override
    {
        coordinator.ThreadDidDie (m_tid, m_error_function);
        return eventLoopResultContinue;
    }

private:

    const lldb::tid_t m_tid;
    ErrorFunction m_error_function;
};

//===----------------------------------------------------------------------===//

class ThreadStateCoordinator::EventRequestResume : public ThreadStateCoordinator::EventBase
{
public:
    EventRequestResume (lldb::tid_t tid,
                        const ThreadIDFunction &request_thread_resume_function,
                        const ErrorFunction &error_function):
    EventBase (),
    m_tid (tid),
    m_request_thread_resume_function (request_thread_resume_function),
    m_error_function (error_function)
    {
    }

    EventLoopResult
    ProcessEvent(ThreadStateCoordinator &coordinator) override
    {
        // Ensure we know about the thread.
        auto find_it = coordinator.m_tid_stop_map.find (m_tid);
        if (find_it == coordinator.m_tid_stop_map.end ())
        {
            // We don't know about this thread.  This is an error condition.
            std::ostringstream error_message;
            error_message << "error: tid " << m_tid << " asked to resume but tid is unknown";
            m_error_function (error_message.str ());
            return eventLoopResultContinue;
        }
        
        // Tell the thread to resume if we don't already think it is running.
        const bool is_stopped = find_it->second;
        if (!is_stopped)
        {
            // Skip the resume call - we have tracked it to be running.
            std::ostringstream error_message;
            error_message << "error: tid " << m_tid << " asked to resume but we think it is already running";
            m_error_function (error_message.str ());
            return eventLoopResultContinue;
        }

        // Before we do the resume below, first check if we have a pending
        // stop notification this is currently or was previously waiting for
        // this thread to stop.  This is potentially a buggy situation since
        // we're ostensibly waiting for threads to stop before we send out the
        // pending notification, and here we are resuming one before we send
        // out the pending stop notification.
        const EventCallAfterThreadsStop *const pending_stop_notification = coordinator.GetPendingThreadStopNotification ();
        if (pending_stop_notification)
        {
            if (pending_stop_notification->GetRemainingWaitTIDs ().count (m_tid) > 0)
            {
                coordinator.Log ("EventRequestResume::%s about to resume tid %" PRIu64 " per explicit request but we have a pending stop notification (tid %" PRIu64 ") that is actively waiting for this thread to stop. Valid sequence of events?", __FUNCTION__, m_tid, pending_stop_notification->GetTriggeringTID ());
            }
            else if (pending_stop_notification->GetInitialWaitTIDs ().count (m_tid) > 0)
            {
                coordinator.Log ("EventRequestResume::%s about to resume tid %" PRIu64 " per explicit request but we have a pending stop notification (tid %" PRIu64 ") that hasn't fired yet and this is one of the threads we had been waiting on (and already marked satisfied for this tid). Valid sequence of events?", __FUNCTION__, m_tid, pending_stop_notification->GetTriggeringTID ());
                for (auto tid : pending_stop_notification->GetRemainingWaitTIDs ())
                {
                    coordinator.Log ("EventRequestResume::%s tid %" PRIu64 " deferred stop notification still waiting on tid  %" PRIu64,
                                     __FUNCTION__,
                                     pending_stop_notification->GetTriggeringTID (),
                                     tid);
                }
            }
        }

        // Request a resume.  We expect this to be synchronous and the system
        // to reflect it is running after this completes.
        m_request_thread_resume_function (m_tid);

        // Now mark it is running.
        find_it->second = false;

        return eventLoopResultContinue;
    }

private:

    const lldb::tid_t m_tid;
    ThreadIDFunction m_request_thread_resume_function;
    ErrorFunction m_error_function;
};

//===----------------------------------------------------------------------===//

ThreadStateCoordinator::ThreadStateCoordinator (const LogFunction &log_function) :
    m_log_function (log_function),
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

    EventCallAfterThreadsStop *const prev_call_after_event = GetPendingThreadStopNotification ();
    if (prev_call_after_event)
    {
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
                                              const ThreadIDFunction &request_thread_stop_function,
                                              const ThreadIDFunction &call_after_function,
                                              const ErrorFunction &error_function)
{
    EnqueueEvent (EventBaseSP (new EventCallAfterThreadsStop (triggering_tid,
                                                              wait_for_stop_tids,
                                                              request_thread_stop_function,
                                                              call_after_function,
                                                              error_function)));
}

void
ThreadStateCoordinator::CallAfterRunningThreadsStop (const lldb::tid_t triggering_tid,
                                                     const ThreadIDFunction &request_thread_stop_function,
                                                     const ThreadIDFunction &call_after_function,
                                                     const ErrorFunction &error_function)
{
    EnqueueEvent (EventBaseSP (new EventCallAfterThreadsStop (triggering_tid,
                                                              request_thread_stop_function,
                                                              call_after_function,
                                                              error_function)));
}

void
ThreadStateCoordinator::ThreadDidStop (lldb::tid_t tid, ErrorFunction &error_function)
{
    // Ensure we know about the thread.
    auto find_it = m_tid_stop_map.find (tid);
    if (find_it == m_tid_stop_map.end ())
    {
        // We don't know about this thread.  This is an error condition.
        std::ostringstream error_message;
        error_message << "error: tid " << tid << " asked to stop but tid is unknown";
        error_function (error_message.str ());
        return;
    }

    // Update the global list of known thread states.  This one is definitely stopped.
    find_it->second = true;

    // If we have a pending notification, remove this from the set.
    EventCallAfterThreadsStop *const call_after_event = GetPendingThreadStopNotification ();
    if (call_after_event)
    {
        const bool pending_stops_remain = call_after_event->RemoveThreadStopRequirementAndMaybeSignal (tid);
        if (!pending_stops_remain)
        {
            // Clear the pending notification now.
            m_pending_notification_sp.reset ();
        }
    }
}

void
ThreadStateCoordinator::ThreadWasCreated (lldb::tid_t tid, bool is_stopped, ErrorFunction &error_function)
{
    // Ensure we don't already know about the thread.
    auto find_it = m_tid_stop_map.find (tid);
    if (find_it != m_tid_stop_map.end ())
    {
        // We already know about this thread.  This is an error condition.
        std::ostringstream error_message;
        error_message << "error: notified tid " << tid << " created but we already know about this thread";
        error_function (error_message.str ());
        return;
    }

    // Add the new thread to the stop map.
    m_tid_stop_map[tid] = is_stopped;

    EventCallAfterThreadsStop *const call_after_event = GetPendingThreadStopNotification ();
    if (call_after_event && !is_stopped)
    {
        // Tell the pending notification that we need to wait
        // for this new thread to stop.
        call_after_event->AddThreadStopRequirement (tid);
    }
}

void
ThreadStateCoordinator::ThreadDidDie (lldb::tid_t tid, ErrorFunction &error_function)
{
    // Ensure we know about the thread.
    auto find_it = m_tid_stop_map.find (tid);
    if (find_it == m_tid_stop_map.end ())
    {
        // We don't know about this thread.  This is an error condition.
        std::ostringstream error_message;
        error_message << "error: notified tid " << tid << " died but tid is unknown";
        error_function (error_message.str ());
        return;
    }

    // Update the global list of known thread states.  While this one is stopped, it is also dead.
    // So stop tracking it.  We assume the user of this coordinator will not keep trying to add
    // dependencies on a thread after it is known to be dead.
    m_tid_stop_map.erase (find_it);

    // If we have a pending notification, remove this from the set.
    EventCallAfterThreadsStop *const call_after_event = GetPendingThreadStopNotification ();
    if (call_after_event)
    {
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

    m_log_function (format, args);

    va_end (args);
}

void
ThreadStateCoordinator::NotifyThreadStop (lldb::tid_t tid,
                                          const ErrorFunction &error_function)
{
    EnqueueEvent (EventBaseSP (new EventThreadStopped (tid, error_function)));
}

void
ThreadStateCoordinator::RequestThreadResume (lldb::tid_t tid,
                                             const ThreadIDFunction &request_thread_resume_function,
                                             const ErrorFunction &error_function)
{
    EnqueueEvent (EventBaseSP (new EventRequestResume (tid, request_thread_resume_function, error_function)));
}

void
ThreadStateCoordinator::NotifyThreadCreate (lldb::tid_t tid,
                                            bool is_stopped,
                                            const ErrorFunction &error_function)
{
    EnqueueEvent (EventBaseSP (new EventThreadCreate (tid, is_stopped, error_function)));
}

void
ThreadStateCoordinator::NotifyThreadDeath (lldb::tid_t tid,
                                           const ErrorFunction &error_function)
{
    EnqueueEvent (EventBaseSP (new EventThreadDeath (tid, error_function)));
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

ThreadStateCoordinator::EventLoopResult
ThreadStateCoordinator::ProcessNextEvent ()
{
    return DequeueEventWithWait()->ProcessEvent (*this);
}

ThreadStateCoordinator::EventCallAfterThreadsStop *
ThreadStateCoordinator::GetPendingThreadStopNotification ()
{
    return static_cast<EventCallAfterThreadsStop*> (m_pending_notification_sp.get ());
}
