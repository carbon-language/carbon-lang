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

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "lldb/lldb-types.h"

namespace lldb_private
{
    class ThreadStateCoordinator
    {
    public:

        // Typedefs.
        typedef std::unordered_set<lldb::tid_t> ThreadIDSet;

        // Protocols.


        // Callback definitions.
        typedef std::function<void (lldb::tid_t tid)> ThreadIDFunc;
        typedef std::function<void (const char *format, va_list args)> LogFunc;

        // constructors
        ThreadStateCoordinator (const LogFunc &log_func);

        // The main purpose of the class: triggering an action after
        // a given set of threads stop.
        void
        CallAfterThreadsStop (lldb::tid_t triggering_tid,
                              const ThreadIDSet &wait_for_stop_tids,
                              const ThreadIDFunc &request_thread_stop_func,
                              const ThreadIDFunc &call_after_func);

        // Notifications called when various state changes occur.
        void
        NotifyThreadStop (lldb::tid_t tid);

        void
        NotifyThreadResume (lldb::tid_t tid);

        void
        NotifyThreadCreate (lldb::tid_t tid);

        void
        NotifyThreadDeath (lldb::tid_t tid);

        // Indicate when the coordinator should shut down.
        void
        StopCoordinator ();

        // Process the next event, returning false when the coordinator is all done.
        // This call is synchronous and blocks when there are no events pending.
        // Expected usage is to run this in a separate thread until the function
        // returns false.  Always call this from the same thread.  The processing
        // logic assumes the execution of this is implicitly serialized.
        bool
        ProcessNextEvent ();

    private:

        // Typedefs.
        class EventBase;

        class EventCallAfterThreadsStop;
        class EventStopCoordinator;
        class EventThreadStopped;

        typedef std::shared_ptr<EventBase> EventBaseSP;

        typedef std::queue<EventBaseSP> QueueType;

        typedef std::unordered_map<lldb::tid_t, bool> TIDBoolMap;


        // Private member functions.
        void
        EnqueueEvent (EventBaseSP event_sp);

        EventBaseSP
        DequeueEventWithWait ();

        void
        SetPendingNotification (const EventBaseSP &event_sp);

        void
        ThreadDidStop (lldb::tid_t tid);

        void
        Log (const char *format, ...);

        // Member variables.
        LogFunc m_log_func;

        QueueType m_event_queue;
        // For now we do simple read/write lock strategy with efficient wait-for-data.
        // We can replace with entirely non-blocking queue later but we still want the
        // reader to sleep when nothing is available - this will be a bursty but infrequent
        // event mechanism.
        std::condition_variable m_queue_condition;
        std::mutex m_queue_mutex;

        EventBaseSP m_pending_notification_sp;

        // Maps known TIDs to stop (true) or not-stopped (false) state.
        TIDBoolMap m_tid_stop_map;
    };
}

#endif
