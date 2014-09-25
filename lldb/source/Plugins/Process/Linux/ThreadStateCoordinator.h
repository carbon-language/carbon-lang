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
#include <unordered_set>

#include "lldb/lldb-types.h"

namespace lldb_private
{
    class ThreadStateCoordinator
    {
    public:

        // Typedefs.
        typedef std::unordered_set<lldb::tid_t> ThreadIDSet;

        // Callback definitions.
        typedef std::function<void (lldb::tid_t tid)> ThreadIDFunc;
        typedef std::function<void (const char *format, va_list args)> LogFunc;

        // constructors
        ThreadStateCoordinator (const LogFunc &log_func);

        // The main purpose of the class: triggering an action after
        // a given set of threads stop.
        void
        CallAfterThreadsStop (const lldb::tid_t triggering_tid,
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
        // returns false.
        bool
        ProcessNextEvent ();

    private:

        enum EventType
        {
            eInvalid,
            eEventTypeCallAfterThreadsStop,
            eEventTypeThreadStopped,
            eEventTypeThreadResumed,
            eEventTypeThreadCreated,
            eEventTypeThreadDied,
        };

        bool m_done_b;

        LogFunc m_log_func;

    };
}

#endif
