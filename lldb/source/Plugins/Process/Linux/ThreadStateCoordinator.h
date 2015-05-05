//===-- ThreadStateCoordinator.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_ThreadStateCoordinator_h
#define lldb_ThreadStateCoordinator_h

#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "lldb/lldb-types.h"

#include "lldb/Core/Error.h"

namespace lldb_private {
namespace process_linux {

    class ThreadStateCoordinator
    {
    public:

        // Typedefs.
        typedef std::unordered_set<lldb::tid_t> ThreadIDSet;

        enum EventLoopResult
        {
            eventLoopResultContinue,
            eventLoopResultStop
        };

        // Callback/block definitions.
        typedef std::function<void (lldb::tid_t tid)> ThreadIDFunction;
        typedef std::function<void (const char *format, va_list args)> LogFunction;
        typedef std::function<void (const std::string &error_message)> ErrorFunction;
        typedef std::function<Error (lldb::tid_t tid)> StopThreadFunction;
        typedef std::function<Error (lldb::tid_t tid, bool supress_signal)> ResumeThreadFunction;

        // Constructors.
        ThreadStateCoordinator (const LogFunction &log_function);

        // Notify the coordinator when a thread is created and/or starting to be
        // tracked.  is_stopped should be true if the thread is currently stopped;
        // otherwise, it should be set false if it is already running.  Will
        // call the error function if the thread id is already tracked.
        void
        NotifyThreadCreate (lldb::tid_t tid,
                            bool is_stopped,
                            const ErrorFunction &error_function);

        // Notify the coordinator when a previously-existing thread should no
        // longer be tracked.  The error_function will trigger if the thread
        // is not being tracked.
        void
        NotifyThreadDeath (lldb::tid_t tid,
                           const ErrorFunction &error_function);


        // This method is the main purpose of the class: triggering a deferred
        // action after a given set of threads stop.  The triggering_tid is the
        // thread id passed to the call_after_function.  The error_function will
        // be fired if either the triggering tid or any of the wait_for_stop_tids
        // are unknown at the time the method is processed.
        void
        CallAfterThreadsStop (lldb::tid_t triggering_tid,
                              const ThreadIDSet &wait_for_stop_tids,
                              const StopThreadFunction &request_thread_stop_function,
                              const ThreadIDFunction &call_after_function,
                              const ErrorFunction &error_function);

        // This method is the main purpose of the class: triggering a deferred
        // action after all non-stopped threads stop.  The triggering_tid is the
        // thread id passed to the call_after_function.  The error_function will
        // be fired if the triggering tid is unknown at the time of execution.
        void
        CallAfterRunningThreadsStop (lldb::tid_t triggering_tid,
                                     const StopThreadFunction &request_thread_stop_function,
                                     const ThreadIDFunction &call_after_function,
                                     const ErrorFunction &error_function);

        // This method is the main purpose of the class: triggering a deferred
        // action after all non-stopped threads stop.  The triggering_tid is the
        // thread id passed to the call_after_function.  The error_function will
        // be fired if the triggering tid is unknown at the time of execution.
        // This variant will send stop requests to all non-stopped threads except
        // for any contained in skip_stop_request_tids.
        void
        CallAfterRunningThreadsStopWithSkipTIDs (lldb::tid_t triggering_tid,
                                                 const ThreadIDSet &skip_stop_request_tids,
                                                 const StopThreadFunction &request_thread_stop_function,
                                                 const ThreadIDFunction &call_after_function,
                                                 const ErrorFunction &error_function);

        // Notify the thread stopped.  Will trigger error at time of execution if we
        // already think it is stopped.
        void
        NotifyThreadStop (lldb::tid_t tid,
                          bool initiated_by_llgs,
                          const ErrorFunction &error_function);

        // Request that the given thread id should have the request_thread_resume_function
        // called.  Will trigger the error_function if the thread is thought to be running
        // already at that point.  This call signals an error if the thread resume is for
        // a thread that is already in a running state.
        void
        RequestThreadResume (lldb::tid_t tid,
                             const ResumeThreadFunction &request_thread_resume_function,
                             const ErrorFunction &error_function);

        // Request that the given thread id should have the request_thread_resume_function
        // called.  Will trigger the error_function if the thread is thought to be running
        // already at that point.  This call ignores threads that are already running and
        // does not trigger an error in that case.
        void
        RequestThreadResumeAsNeeded (lldb::tid_t tid,
                                     const ResumeThreadFunction &request_thread_resume_function,
                                     const ErrorFunction &error_function);

        // Indicate the calling process did an exec and that the thread state
        // should be 100% cleared.
        void
        ResetForExec ();

        // Enable/disable verbose logging of event processing.
        void
        LogEnableEventProcessing (bool enabled);

    private:

        // Typedefs.
        class EventBase;

        class EventCallAfterThreadsStop;
        class EventThreadStopped;
        class EventThreadCreate;
        class EventThreadDeath;
        class EventRequestResume;

        class EventReset;

        typedef std::shared_ptr<EventBase> EventBaseSP;

        enum class ThreadState
        {
            Running,
            Stopped
        };

        struct ThreadContext
        {
            ThreadState m_state;
            bool m_stop_requested = false;
            ResumeThreadFunction m_request_resume_function;
        };
        typedef std::unordered_map<lldb::tid_t, ThreadContext> TIDContextMap;


        std::mutex m_event_mutex; // Serializes execution of ProcessEvent.

        void
        ProcessEvent (const EventBaseSP &event_sp);

        void
        SetPendingNotification (const EventBaseSP &event_sp);

        void
        ThreadDidStop (lldb::tid_t tid, bool initiated_by_llgs, ErrorFunction &error_function);

        void
        ThreadWasCreated (lldb::tid_t tid, bool is_stopped, ErrorFunction &error_function);

        void
        ThreadDidDie (lldb::tid_t tid, ErrorFunction &error_function);

        void
        ResetNow ();

        bool
        IsKnownThread(lldb::tid_t tid) const;

        void
        Log (const char *format, ...);

        EventCallAfterThreadsStop *
        GetPendingThreadStopNotification ();

        // Member variables.
        LogFunction m_log_function;

        EventBaseSP m_pending_notification_sp;

        // Maps known TIDs to ThreadContext.
        TIDContextMap m_tid_map;

        bool m_log_event_processing;
    };

} // namespace process_linux
} // namespace lldb_private

#endif
