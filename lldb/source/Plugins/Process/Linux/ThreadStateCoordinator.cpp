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
using namespace lldb_private::process_linux;

//===----------------------------------------------------------------------===//

void
ThreadStateCoordinator::DoResume(
        lldb::tid_t tid,
        ResumeThreadFunction request_thread_resume_function,
        ErrorFunction error_function,
        bool error_when_already_running)
{
    // Ensure we know about the thread.
    auto find_it = m_tid_map.find (tid);
    if (find_it == m_tid_map.end ())
    {
        // We don't know about this thread.  This is an error condition.
        std::ostringstream error_message;
        error_message << "error: tid " << tid << " asked to resume but tid is unknown";
        error_function (error_message.str ());
        return;
    }
    auto& context = find_it->second;
    // Tell the thread to resume if we don't already think it is running.
    const bool is_stopped = context.m_state == ThreadState::Stopped;
    if (!is_stopped)
    {
        // It's not an error, just a log, if the error_when_already_running flag is not set.
        // This covers cases where, for instance, we're just trying to resume all threads
        // from the user side.
        if (!error_when_already_running)
        {
            Log ("ThreadStateCoordinator::%s tid %" PRIu64 " optional resume skipped since it is already running",
                             __FUNCTION__,
                             tid);
        }
        else
        {
            // Skip the resume call - we have tracked it to be running.  And we unconditionally
            // expected to resume this thread.  Flag this as an error.
            std::ostringstream error_message;
            error_message << "error: tid " << tid << " asked to resume but we think it is already running";
            error_function (error_message.str ());
        }

        // Error or not, we're done.
        return;
    }

    // Before we do the resume below, first check if we have a pending
    // stop notification this is currently or was previously waiting for
    // this thread to stop.  This is potentially a buggy situation since
    // we're ostensibly waiting for threads to stop before we send out the
    // pending notification, and here we are resuming one before we send
    // out the pending stop notification.
    if (m_pending_notification_up)
    {
        if (m_pending_notification_up->wait_for_stop_tids.count (tid) > 0)
        {
            Log ("ThreadStateCoordinator::%s about to resume tid %" PRIu64 " per explicit request but we have a pending stop notification (tid %" PRIu64 ") that is actively waiting for this thread to stop. Valid sequence of events?", __FUNCTION__, tid, m_pending_notification_up->triggering_tid);
        }
        else if (m_pending_notification_up->original_wait_for_stop_tids.count (tid) > 0)
        {
            Log ("ThreadStateCoordinator::%s about to resume tid %" PRIu64 " per explicit request but we have a pending stop notification (tid %" PRIu64 ") that hasn't fired yet and this is one of the threads we had been waiting on (and already marked satisfied for this tid). Valid sequence of events?", __FUNCTION__, tid, m_pending_notification_up->triggering_tid);
            for (auto tid : m_pending_notification_up->wait_for_stop_tids)
            {
                Log ("ThreadStateCoordinator::%s tid %" PRIu64 " deferred stop notification still waiting on tid  %" PRIu64,
                                 __FUNCTION__,
                                 m_pending_notification_up->triggering_tid,
                                 tid);
            }
        }
    }

    // Request a resume.  We expect this to be synchronous and the system
    // to reflect it is running after this completes.
    const auto error = request_thread_resume_function (tid, false);
    if (error.Success ())
    {
        // Now mark it is running.
        context.m_state = ThreadState::Running;
        context.m_request_resume_function = request_thread_resume_function;
    }
    else
    {
        Log ("ThreadStateCoordinator::%s failed to resume thread tid  %" PRIu64 ": %s",
                         __FUNCTION__, tid, error.AsCString ());
    }

    return;
}

//===----------------------------------------------------------------------===//

ThreadStateCoordinator::ThreadStateCoordinator (const LogFunction &log_function) :
    m_log_function (log_function),
    m_tid_map (),
    m_log_event_processing (false)
{
}

void
ThreadStateCoordinator::CallAfterThreadsStop (const lldb::tid_t triggering_tid,
                                              const ThreadIDSet &wait_for_stop_tids,
                                              const StopThreadFunction &request_thread_stop_function,
                                              const ThreadIDFunction &call_after_function,
                                              const ErrorFunction &error_function)
{
    std::lock_guard<std::mutex> lock(m_event_mutex);

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s about to process event: (triggering_tid: %" PRIu64 ", wait_for_stop_tids.size(): %zd)",
                __FUNCTION__, triggering_tid, wait_for_stop_tids.size());
    }

    DoCallAfterThreadsStop(PendingNotificationUP(new PendingNotification(
                triggering_tid, wait_for_stop_tids, request_thread_stop_function,
                call_after_function, error_function)));

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s event processing done", __FUNCTION__);
    }
}

void
ThreadStateCoordinator::CallAfterRunningThreadsStop (const lldb::tid_t triggering_tid,
                                                     const StopThreadFunction &request_thread_stop_function,
                                                     const ThreadIDFunction &call_after_function,
                                                     const ErrorFunction &error_function)
{
    std::lock_guard<std::mutex> lock(m_event_mutex);

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s about to process event: (triggering_tid: %" PRIu64 ")",
                __FUNCTION__, triggering_tid);
    }

    DoCallAfterThreadsStop(PendingNotificationUP(new PendingNotification(
                triggering_tid,
                request_thread_stop_function,
                call_after_function,
                error_function)));

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s event processing done", __FUNCTION__);
    }
}

void
ThreadStateCoordinator::CallAfterRunningThreadsStopWithSkipTIDs (lldb::tid_t triggering_tid,
                                                                 const ThreadIDSet &skip_stop_request_tids,
                                                                 const StopThreadFunction &request_thread_stop_function,
                                                                 const ThreadIDFunction &call_after_function,
                                                                 const ErrorFunction &error_function)
{
    std::lock_guard<std::mutex> lock(m_event_mutex);

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s about to process event: (triggering_tid: %" PRIu64 ", skip_stop_request_tids.size(): %zd)",
                __FUNCTION__, triggering_tid, skip_stop_request_tids.size());
    }

    DoCallAfterThreadsStop(PendingNotificationUP(new PendingNotification(
                triggering_tid,
                request_thread_stop_function,
                call_after_function,
                skip_stop_request_tids,
                error_function)));

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s event processing done", __FUNCTION__);
    }
}

void
ThreadStateCoordinator::SignalIfRequirementsSatisfied()
{
    if (m_pending_notification_up && m_pending_notification_up->wait_for_stop_tids.empty ())
    {
        m_pending_notification_up->call_after_function(m_pending_notification_up->triggering_tid);
        m_pending_notification_up.reset();
    }
}

bool
ThreadStateCoordinator::RequestStopOnAllSpecifiedThreads()
{
    // Request a stop for all the thread stops that need to be stopped
    // and are not already known to be stopped.  Keep a list of all the
    // threads from which we still need to hear a stop reply.

    ThreadIDSet sent_tids;
    for (auto tid : m_pending_notification_up->wait_for_stop_tids)
    {
        // Validate we know about all tids for which we must first receive a stop before
        // triggering the deferred stop notification.
        auto find_it = m_tid_map.find (tid);
        if (find_it == m_tid_map.end ())
        {
            // This is an error.  We shouldn't be asking for waiting pids that aren't known.
            // NOTE: we may be stripping out the specification of wait tids and handle this
            // automatically, in which case this state can never occur.
            std::ostringstream error_message;
            error_message << "error: deferred notification for tid " << m_pending_notification_up->triggering_tid << " specified an unknown/untracked pending stop tid " << m_pending_notification_up->triggering_tid;
            m_pending_notification_up->error_function (error_message.str ());

            // Bail out here.
            return false;
        }

        // If the pending stop thread is currently running, we need to send it a stop request.
        auto& context = find_it->second;
        if (context.m_state == ThreadState::Running)
        {
            RequestThreadStop (tid, context);
            sent_tids.insert (tid);
        }
    }
    // We only need to wait for the sent_tids - so swap our wait set
    // to the sent tids.  The rest are already stopped and we won't
    // be receiving stop notifications for them.
    m_pending_notification_up->wait_for_stop_tids.swap (sent_tids);

    // Succeeded, keep running.
    return true;
}

void
ThreadStateCoordinator::RequestStopOnAllRunningThreads()
{
    // Request a stop for all the thread stops that need to be stopped
    // and are not already known to be stopped.  Keep a list of all the
    // threads from which we still need to hear a stop reply.

    ThreadIDSet sent_tids;
    for (auto it = m_tid_map.begin(); it != m_tid_map.end(); ++it)
    {
        // We only care about threads not stopped.
        const bool running = it->second.m_state == ThreadState::Running;
        if (running)
        {
            const lldb::tid_t tid = it->first;

            // Request this thread stop if the tid stop request is not explicitly ignored.
            const bool skip_stop_request = m_pending_notification_up->skip_stop_request_tids.count (tid) > 0;
            if (!skip_stop_request)
                RequestThreadStop (tid, it->second);

            // Even if we skipped sending the stop request for other reasons (like stepping),
            // we still need to wait for that stepping thread to notify completion/stop.
            sent_tids.insert (tid);
        }
    }

    // Set the wait list to the set of tids for which we requested stops.
    m_pending_notification_up->wait_for_stop_tids.swap (sent_tids);
}

void
ThreadStateCoordinator::RequestThreadStop (lldb::tid_t tid, ThreadContext& context)
{
    const auto error = m_pending_notification_up->request_thread_stop_function (tid);
    if (error.Success ())
        context.m_stop_requested = true;
    else
    {
        Log ("ThreadStateCoordinator::%s failed to request thread stop tid  %" PRIu64 ": %s",
                         __FUNCTION__, tid, error.AsCString ());
    }
}


void
ThreadStateCoordinator::ThreadDidStop (lldb::tid_t tid, bool initiated_by_llgs, const ErrorFunction &error_function)
{
    // Ensure we know about the thread.
    auto find_it = m_tid_map.find (tid);
    if (find_it == m_tid_map.end ())
    {
        // We don't know about this thread.  This is an error condition.
        std::ostringstream error_message;
        error_message << "error: tid " << tid << " asked to stop but tid is unknown";
        error_function (error_message.str ());
        return;
    }

    // Update the global list of known thread states.  This one is definitely stopped.
    auto& context = find_it->second;
    const auto stop_was_requested = context.m_stop_requested;
    context.m_state = ThreadState::Stopped;
    context.m_stop_requested = false;

    // If we have a pending notification, remove this from the set.
    if (m_pending_notification_up)
    {
        m_pending_notification_up->wait_for_stop_tids.erase(tid);
        SignalIfRequirementsSatisfied();
    }

    if (initiated_by_llgs && context.m_request_resume_function && !stop_was_requested)
    {
        // We can end up here if stop was initiated by LLGS but by this time a
        // thread stop has occurred - maybe initiated by another event.
        Log ("Resuming thread %"  PRIu64 " since stop wasn't requested", tid);
        const auto error = context.m_request_resume_function (tid, true);
        if (error.Success ())
        {
            context.m_state = ThreadState::Running;
        }
        else
        {
            Log ("ThreadStateCoordinator::%s failed to resume thread tid  %" PRIu64 ": %s",
                 __FUNCTION__, tid, error.AsCString ());
        }
    }
}

void
ThreadStateCoordinator::DoCallAfterThreadsStop(PendingNotificationUP &&notification_up)
{
    // Validate we know about the deferred trigger thread.
    if (!IsKnownThread (notification_up->triggering_tid))
    {
        // We don't know about this thread.  This is an error condition.
        std::ostringstream error_message;
        error_message << "error: deferred notification tid " << notification_up->triggering_tid << " is unknown";
        notification_up->error_function (error_message.str ());

        // We bail out here.
        return;
    }

    if (m_pending_notification_up)
    {
        // Yikes - we've already got a pending signal notification in progress.
        // Log this info.  We lose the pending notification here.
        Log ("ThreadStateCoordinator::%s dropping existing pending signal notification for tid %" PRIu64 ", to be replaced with signal for tid %" PRIu64,
                   __FUNCTION__,
                   m_pending_notification_up->triggering_tid,
                   notification_up->triggering_tid);
    }
    m_pending_notification_up = std::move(notification_up);

    if (m_pending_notification_up->request_stop_on_all_unstopped_threads)
        RequestStopOnAllRunningThreads();
    else
    {
        if (!RequestStopOnAllSpecifiedThreads())
            return;
    }

    if (m_pending_notification_up->wait_for_stop_tids.empty ())
    {
        // We're not waiting for any threads.  Fire off the deferred signal delivery event.
        m_pending_notification_up->call_after_function(m_pending_notification_up->triggering_tid);
        m_pending_notification_up.reset();
    }
}

void
ThreadStateCoordinator::ThreadWasCreated (lldb::tid_t tid, bool is_stopped, const ErrorFunction &error_function)
{
    // Ensure we don't already know about the thread.
    auto find_it = m_tid_map.find (tid);
    if (find_it != m_tid_map.end ())
    {
        // We already know about this thread.  This is an error condition.
        std::ostringstream error_message;
        error_message << "error: notified tid " << tid << " created but we already know about this thread";
        error_function (error_message.str ());
        return;
    }

    // Add the new thread to the stop map.
    ThreadContext ctx;
    ctx.m_state = (is_stopped) ? ThreadState::Stopped : ThreadState::Running;
    m_tid_map[tid] = std::move(ctx);

    if (m_pending_notification_up && !is_stopped)
    {
        // We will need to wait for this new thread to stop as well before firing the
        // notification.
        m_pending_notification_up->wait_for_stop_tids.insert(tid);
        m_pending_notification_up->request_thread_stop_function(tid);
    }
}

void
ThreadStateCoordinator::ThreadDidDie (lldb::tid_t tid, const ErrorFunction &error_function)
{
    // Ensure we know about the thread.
    auto find_it = m_tid_map.find (tid);
    if (find_it == m_tid_map.end ())
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
    m_tid_map.erase (find_it);

    // If we have a pending notification, remove this from the set.
    if (m_pending_notification_up)
    {
        m_pending_notification_up->wait_for_stop_tids.erase(tid);
        SignalIfRequirementsSatisfied();
    }
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
                                          bool initiated_by_llgs,
                                          const ErrorFunction &error_function)
{
    std::lock_guard<std::mutex> lock(m_event_mutex);

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s about to process event: (tid: %" PRIu64 ", %sinitiated by llgs)",
                __FUNCTION__, tid, initiated_by_llgs?"":"not ");
    }

    ThreadDidStop (tid, initiated_by_llgs, error_function);

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s event processing done", __FUNCTION__);
    }
}

void
ThreadStateCoordinator::RequestThreadResume (lldb::tid_t tid,
                                             const ResumeThreadFunction &request_thread_resume_function,
                                             const ErrorFunction &error_function)
{
    std::lock_guard<std::mutex> lock(m_event_mutex);

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s about to process event: (tid: %" PRIu64 ")",
                __FUNCTION__, tid);
    }

    DoResume(tid, request_thread_resume_function, error_function, true);

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s event processing done", __FUNCTION__);
    }
}

void
ThreadStateCoordinator::RequestThreadResumeAsNeeded (lldb::tid_t tid,
                                                     const ResumeThreadFunction &request_thread_resume_function,
                                                     const ErrorFunction &error_function)
{
    std::lock_guard<std::mutex> lock(m_event_mutex);

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s about to process event: (tid: %" PRIu64 ")",
                __FUNCTION__, tid);
    }

    DoResume (tid, request_thread_resume_function, error_function, false);

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s event processing done", __FUNCTION__);
    }
}

void
ThreadStateCoordinator::NotifyThreadCreate (lldb::tid_t tid,
                                            bool is_stopped,
                                            const ErrorFunction &error_function)
{
    std::lock_guard<std::mutex> lock(m_event_mutex);

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s about to process event: (tid: %" PRIu64 ", is %sstopped)",
                __FUNCTION__, tid, is_stopped?"":"not ");
    }

    ThreadWasCreated (tid, is_stopped, error_function);

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s event processing done", __FUNCTION__);
    }
}

void
ThreadStateCoordinator::NotifyThreadDeath (lldb::tid_t tid,
                                           const ErrorFunction &error_function)
{
    std::lock_guard<std::mutex> lock(m_event_mutex);

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s about to process event: (tid: %" PRIu64 ")", __FUNCTION__, tid);
    }

    ThreadDidDie(tid, error_function);

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s event processing done", __FUNCTION__);
    }
}

void
ThreadStateCoordinator::ResetForExec ()
{
    std::lock_guard<std::mutex> lock(m_event_mutex);

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s about to process event", __FUNCTION__);
    }

    // Clear the pending notification if there was one.
    m_pending_notification_up.reset ();

    // Clear the stop map - we no longer know anything about any thread state.
    // The caller is expected to reset thread states for all threads, and we
    // will assume anything we haven't heard about is running and requires a
    // stop.
    m_tid_map.clear ();

    if (m_log_event_processing)
    {
        Log ("ThreadStateCoordinator::%s event processing done", __FUNCTION__);
    }
}
void
ThreadStateCoordinator::LogEnableEventProcessing (bool enabled)
{
    m_log_event_processing = enabled;
}

bool
ThreadStateCoordinator::IsKnownThread (lldb::tid_t tid) const
{
    return m_tid_map.find (tid) != m_tid_map.end ();
}
