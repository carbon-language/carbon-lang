#include <limits.h>
#include "gtest/gtest.h"

#include "Plugins/Process/Linux/ThreadStateCoordinator.h"

using namespace lldb_private;

namespace
{
    const ThreadStateCoordinator::ThreadIDSet EMPTY_THREAD_ID_SET;

    void
    NOPLogger (const char *format, va_list args)
    {
        // Do nothing.
    }

    void
    StdoutLogger (const char *format, va_list args)
    {
        // Print to stdout.
        vprintf (format, args);
        printf ("\n");
    }

    // These are single-line macros so that IDE integration of gtest results puts
    // the error markers on the correct failure point within the gtest.

#define ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS() ASSERT_EQ (ThreadStateCoordinator::eventLoopResultContinue, m_coordinator.ProcessNextEvent ()); \
if (HasError ()) { printf ("unexpected error in processing of event, error: %s\n", m_error_string.c_str ()); } \
ASSERT_EQ (false, HasError ());

#define ASSERT_PROCESS_NEXT_EVENT_FAILS() ASSERT_EQ (ThreadStateCoordinator::eventLoopResultContinue, m_coordinator.ProcessNextEvent ()); \
ASSERT_EQ (true, HasError ());

    class ThreadStateCoordinatorTest: public ::testing::Test
    {
    protected:
        // Constants.
        const lldb::tid_t TRIGGERING_TID      = 4105;
        const lldb::tid_t PENDING_STOP_TID    = 3;
        const lldb::tid_t PENDING_STOP_TID_02 = 29016;
        const lldb::tid_t NEW_THREAD_TID      = 1234;

        // Member variables.
        bool m_error_called = false;
        std::string m_error_string;

        ThreadStateCoordinator m_coordinator;

        bool m_deferred_notification_called;
        lldb::tid_t m_deferred_notification_tid;

        ThreadStateCoordinator::ThreadIDSet m_requested_stop_tids;

        // Constructors.
        ThreadStateCoordinatorTest () :
        m_error_called (false),
        m_error_string (),
        m_coordinator (StdoutLogger),
        m_deferred_notification_called (false),
        m_deferred_notification_tid (0),
        m_requested_stop_tids ()
        {
        }

        // Member functions.

        // Error handling.
        void
        SetErrorString (const std::string &error_string)
        {
            m_error_called = true;
            m_error_string = error_string;
            printf ("received error: %s (test might be expecting)\n", error_string.c_str ());
        }

        ThreadStateCoordinator::ErrorFunction
        GetErrorFunction ()
        {
            using namespace std::placeholders;
            return std::bind(&ThreadStateCoordinatorTest::SetErrorString, this, _1);
        }

        bool
        HasError () const
        {
            return m_error_called;
        }

        // Deferred notificaton reception.
        void
        DeferredStopNotificationHandler (lldb::tid_t triggered_tid)
        {
            m_deferred_notification_called = true;
            m_deferred_notification_tid = triggered_tid;
        }

        ThreadStateCoordinator::ThreadIDFunction
        GetDeferredStopNotificationFunction ()
        {
            using namespace std::placeholders;
            return std::bind(&ThreadStateCoordinatorTest::DeferredStopNotificationHandler, this, _1);
        }

        bool
        DidFireDeferredNotification () const
        {
            return m_deferred_notification_called;
        }

        lldb::tid_t
        GetDeferredNotificationTID () const
        {
            return m_deferred_notification_tid;
        }

        // Stop request call reception.
        void
        ThreadStopRequested (lldb::tid_t stop_tid)
        {
            m_requested_stop_tids.insert (stop_tid);
        }

        ThreadStateCoordinator::ThreadIDFunction
        GetStopRequestFunction ()
        {
            using namespace std::placeholders;
            return std::bind(&ThreadStateCoordinatorTest::ThreadStopRequested, this, _1);
        }

        ThreadStateCoordinator::ThreadIDSet::size_type
        GetRequestedStopCount () const
        {
            return m_requested_stop_tids.size();
        }

        bool
        DidRequestStopForTid (lldb::tid_t tid)
        {
            return m_requested_stop_tids.find (tid) != m_requested_stop_tids.end ();
        }

        // Test state initialization helpers.
        void
        SetupKnownRunningThread (lldb::tid_t tid)
        {
            NotifyThreadCreate (tid, false);
            ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();
        }

        void
        SetupKnownStoppedThread (lldb::tid_t tid)
        {
            NotifyThreadCreate (tid, true);
            ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();
        }

        // Convenience wrappers for ThreadStateCoordinator, using defaults for expected arguments
        // that plug into the test case handlers.
        void
        CallAfterThreadsStop (lldb::tid_t deferred_tid,
                              const ThreadStateCoordinator::ThreadIDSet &pending_stop_wait_tids)
        {
            m_coordinator.CallAfterThreadsStop (deferred_tid,
                                                pending_stop_wait_tids,
                                                GetStopRequestFunction (),
                                                GetDeferredStopNotificationFunction (),
                                                GetErrorFunction ());
        }

        void
        CallAfterRunningThreadsStop (lldb::tid_t deferred_tid)
        {
            m_coordinator.CallAfterRunningThreadsStop (deferred_tid,
                                                       GetStopRequestFunction (),
                                                       GetDeferredStopNotificationFunction (),
                                                       GetErrorFunction ());
        }

        void
        NotifyThreadCreate (lldb::tid_t stopped_tid, bool thread_is_stopped)
        {
            m_coordinator.NotifyThreadCreate (stopped_tid, thread_is_stopped, GetErrorFunction ());
        }

        void
        NotifyThreadStop (lldb::tid_t stopped_tid)
        {
            m_coordinator.NotifyThreadStop (stopped_tid, GetErrorFunction ());
        }

        void
        NotifyThreadDeath (lldb::tid_t tid)
        {
            m_coordinator.NotifyThreadDeath (tid, GetErrorFunction ());
        }
};
}

TEST_F (ThreadStateCoordinatorTest, StopCoordinatorWorksNoPriorEvents)
{
    m_coordinator.StopCoordinator ();
    ASSERT_EQ (ThreadStateCoordinator::eventLoopResultStop, m_coordinator.ProcessNextEvent ());
    ASSERT_EQ (false, HasError ());
}

TEST_F (ThreadStateCoordinatorTest, NotifyThreadStopSignalsErrorOnUnknownThread)
{
    const lldb::tid_t UNKNOWN_TID = 678;

    // Notify an unknown thread has stopped.
    NotifyThreadStop (UNKNOWN_TID);
    ASSERT_PROCESS_NEXT_EVENT_FAILS ();
}

TEST_F (ThreadStateCoordinatorTest, CallAfterTheadsStopSignalsErrorOnUnknownDeferredThread)
{
    const lldb::tid_t UNKNOWN_TRIGGER_TID = 678;

    // Defer notify for an unknown thread.
    CallAfterThreadsStop (UNKNOWN_TRIGGER_TID,
                          EMPTY_THREAD_ID_SET);

    // Shouldn't have fired yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Event should fail because trigger tid is unknown.
    ASSERT_PROCESS_NEXT_EVENT_FAILS ();

    // Shouldn't have fired due to error.
    ASSERT_EQ (false, DidFireDeferredNotification ());
}

TEST_F (ThreadStateCoordinatorTest, CallAfterTheadsStopSignalsErrorOnUnknownPendingStopThread)
{
    // Let the coordinator know about our thread.
    SetupKnownStoppedThread (TRIGGERING_TID);

    // Defer notify for an unknown thread.
    const lldb::tid_t UNKNOWN_PENDING_STOP_TID = 7890;
    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { UNKNOWN_PENDING_STOP_TID };

    CallAfterThreadsStop (TRIGGERING_TID,
                          pending_stop_tids);

    // Shouldn't have fired yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Event should fail because trigger tid is unknown.
    ASSERT_PROCESS_NEXT_EVENT_FAILS ();

    // Shouldn't have triggered deferred notification due to error.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Shouldn't have triggered stop request due to unknown tid.
    ASSERT_EQ (0, GetRequestedStopCount ());
}

TEST_F (ThreadStateCoordinatorTest, CallAfterThreadsStopFiresWhenNoPendingStops)
{
    // Let the coordinator know about our thread.
    SetupKnownStoppedThread (TRIGGERING_TID);

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    CallAfterThreadsStop (TRIGGERING_TID,
                          EMPTY_THREAD_ID_SET);

    // Notification trigger shouldn't go off yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Process next event.  This will pick up the call after threads stop event.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Now the trigger should have fired, since there were no threads that needed to first stop.
    ASSERT_EQ (true, DidFireDeferredNotification ());
    ASSERT_EQ (TRIGGERING_TID, GetDeferredNotificationTID ());
}

TEST_F (ThreadStateCoordinatorTest, CallAfterThreadsStopFiresWhenOnePendingStop)
{
    // Let the coordinator know about our thread.
    SetupKnownStoppedThread (TRIGGERING_TID);

    // Let the coordinator know about a currently-running thread we'll wait on.
    SetupKnownRunningThread (PENDING_STOP_TID);

    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { PENDING_STOP_TID };

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    CallAfterThreadsStop (TRIGGERING_TID,
                          pending_stop_tids);

    // Neither trigger should have gone off yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());
    ASSERT_EQ (false, DidRequestStopForTid (PENDING_STOP_TID));

    // Process next event.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Now the request thread stop should have been called for the pending stop.
    ASSERT_EQ (true, DidRequestStopForTid (PENDING_STOP_TID));

    // But we still shouldn't have the deferred signal call go off yet.  Need to wait for the stop to be reported.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Now report the that the pending stop occurred.
    NotifyThreadStop (PENDING_STOP_TID);

    // Shouldn't take effect until after next processing step.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Process next event.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Deferred signal notification should have fired now.
    ASSERT_EQ (true, DidFireDeferredNotification ());
    ASSERT_EQ (TRIGGERING_TID, GetDeferredNotificationTID ());
}

TEST_F (ThreadStateCoordinatorTest, CallAfterThreadsStopFiresWhenTwoPendingStops)
{
    // Setup threads.
    SetupKnownStoppedThread (TRIGGERING_TID);
    SetupKnownRunningThread (PENDING_STOP_TID);
    SetupKnownRunningThread (PENDING_STOP_TID_02);

    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { PENDING_STOP_TID, PENDING_STOP_TID_02 };

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    CallAfterThreadsStop (TRIGGERING_TID,
                          pending_stop_tids);

    // Neither trigger should have gone off yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());
    ASSERT_EQ (0, GetRequestedStopCount ());

    // Process next event.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Now the request thread stops should have been called for the pending stop tids.
    ASSERT_EQ (2, GetRequestedStopCount ());
    ASSERT_EQ (true, DidRequestStopForTid (PENDING_STOP_TID));
    ASSERT_EQ (true, DidRequestStopForTid (PENDING_STOP_TID_02));

    // But we still shouldn't have the deferred signal call go off yet.  Need to wait for the stop to be reported.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Report the that the first pending stop occurred.
    NotifyThreadStop (PENDING_STOP_TID);
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Shouldn't take effect until after both pending threads are notified.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Report the that the first pending stop occurred.
    NotifyThreadStop (PENDING_STOP_TID_02);
    ASSERT_EQ (false, DidFireDeferredNotification ());
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Deferred signal notification should have fired now.
    ASSERT_EQ (true, DidFireDeferredNotification ());
    ASSERT_EQ (TRIGGERING_TID, GetDeferredNotificationTID ());
}

TEST_F (ThreadStateCoordinatorTest, CallAfterThreadsStopFiresWhenPendingAlreadyStopped)
{
    // Setup threads.
    SetupKnownStoppedThread (TRIGGERING_TID);
    SetupKnownRunningThread (PENDING_STOP_TID);

    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { PENDING_STOP_TID };

    // Tell m_coordinator the pending stop tid is already stopped.
    NotifyThreadStop (PENDING_STOP_TID);
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    CallAfterThreadsStop (TRIGGERING_TID,
                          pending_stop_tids);

    // Neither trigger should have gone off yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());
    ASSERT_EQ (false, DidRequestStopForTid (PENDING_STOP_TID));

    // Process next event.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // The pending stop should *not* fire because the m_coordinator knows it has already stopped.
    ASSERT_EQ (false, DidRequestStopForTid (PENDING_STOP_TID));

    // The deferred signal notification should have fired since all requirements were met.
    ASSERT_EQ (true, DidFireDeferredNotification ());
    ASSERT_EQ (TRIGGERING_TID, GetDeferredNotificationTID ());
}

TEST_F (ThreadStateCoordinatorTest, CallAfterThreadsStopFiresWhenTwoPendingOneAlreadyStopped)
{
    SetupKnownStoppedThread (TRIGGERING_TID);
    SetupKnownRunningThread (PENDING_STOP_TID);
    SetupKnownRunningThread (PENDING_STOP_TID_02);

    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { PENDING_STOP_TID, PENDING_STOP_TID_02 };

    // Tell coordinator the pending stop tid is already stopped.
    NotifyThreadStop (PENDING_STOP_TID);
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    CallAfterThreadsStop (TRIGGERING_TID,
                          pending_stop_tids);

    // Neither trigger should have gone off yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());
    ASSERT_EQ (0, GetRequestedStopCount ());

    // Process next event.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // The pending stop should only fire for one of the threads, the one that wasn't already stopped.
    ASSERT_EQ (true, DidRequestStopForTid (PENDING_STOP_TID_02));
    ASSERT_EQ (false, DidRequestStopForTid (PENDING_STOP_TID));

    // The deferred signal notification should not yet have fired since all pending thread stops have not yet occurred.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Notify final thread has stopped.
    NotifyThreadStop (PENDING_STOP_TID_02);
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // The deferred signal notification should have fired since all requirements were met.
    ASSERT_EQ (true, DidFireDeferredNotification ());
    ASSERT_EQ (TRIGGERING_TID, GetDeferredNotificationTID ());
}

TEST_F (ThreadStateCoordinatorTest, CallAfterThreadsStopFiresWhenOnePendingThreadDies)
{
    SetupKnownStoppedThread (TRIGGERING_TID);
    SetupKnownRunningThread (PENDING_STOP_TID);

    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { PENDING_STOP_TID };

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    CallAfterThreadsStop (TRIGGERING_TID,
                          pending_stop_tids);

    // Neither trigger should have gone off yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());
    ASSERT_EQ (0, GetRequestedStopCount ());

    // Process next event.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Now the request thread stop should have been called for the pending stop.
    ASSERT_EQ (true, DidRequestStopForTid (PENDING_STOP_TID));

    // But we still shouldn't have the deferred signal call go off yet.  Need to wait for the death to be reported.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Now report the that the thread with pending stop dies.
    NotifyThreadDeath (PENDING_STOP_TID);

    // Shouldn't take effect until after next processing step.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Process next event.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Deferred signal notification should have fired now.
    ASSERT_EQ (true, DidFireDeferredNotification ());
    ASSERT_EQ (TRIGGERING_TID, GetDeferredNotificationTID ());
}

TEST_F (ThreadStateCoordinatorTest, ExistingPendingNotificationRequiresStopFromNewThread)
{
    SetupKnownStoppedThread (TRIGGERING_TID);
    SetupKnownRunningThread (PENDING_STOP_TID);

    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { PENDING_STOP_TID };

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    CallAfterThreadsStop (TRIGGERING_TID,
                          pending_stop_tids);

    // Neither trigger should have gone off yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());
    ASSERT_EQ (0, GetRequestedStopCount ());

    // Process next event.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Now the request thread stop should have been called for the pending stop.
    ASSERT_EQ (true, DidRequestStopForTid (PENDING_STOP_TID));

    // But we still shouldn't have the deferred signal call go off yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Indicate a new thread has just been created.
    SetupKnownRunningThread (NEW_THREAD_TID);

    // We should have just received a stop request for the new thread id.
    ASSERT_EQ (2, GetRequestedStopCount ());
    ASSERT_EQ (true, DidRequestStopForTid (NEW_THREAD_TID));

    // Now report the original pending tid stopped.  This should no longer
    // trigger the pending notification because we should now require the
    // new thread to stop too.
    NotifyThreadStop (PENDING_STOP_TID);
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Now notify the new thread stopped.
    NotifyThreadStop (NEW_THREAD_TID);
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Deferred signal notification should have fired now.
    ASSERT_EQ (true, DidFireDeferredNotification ());
    ASSERT_EQ (TRIGGERING_TID, GetDeferredNotificationTID ());
}

TEST_F (ThreadStateCoordinatorTest, DeferredNotificationRemovedByResetForExec)
{
    SetupKnownStoppedThread (TRIGGERING_TID);

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    CallAfterThreadsStop (TRIGGERING_TID,
                          EMPTY_THREAD_ID_SET);

    // Notification trigger shouldn't go off yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Now indicate an exec occurred, which will invalidate all state about the process and threads.
    m_coordinator.ResetForExec ();

    // Verify the deferred stop notification does *not* fire with the next
    // process.  It will handle the reset and not the deferred signaling, which
    // should now be removed.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();
    ASSERT_EQ (false, DidFireDeferredNotification ());
}

TEST_F (ThreadStateCoordinatorTest, RequestThreadResumeSignalsErrorOnUnknownThread)
{
    const lldb::tid_t UNKNOWN_TID = 411;

    // Request a resume.
    lldb::tid_t resumed_tid = 0;
    int resume_call_count = 0;

    m_coordinator.RequestThreadResume (UNKNOWN_TID,
                                       [&](lldb::tid_t tid)
                                       {
                                           ++resume_call_count;
                                           resumed_tid = tid;
                                       },
                                       GetErrorFunction ());
    // Shouldn't be called yet.
    ASSERT_EQ (0, resume_call_count);

    // Process next event.  This should fail since the coordinator doesn't know about the thread.
    ASSERT_PROCESS_NEXT_EVENT_FAILS ();
    ASSERT_EQ (0, resume_call_count);
}

TEST_F (ThreadStateCoordinatorTest, RequestThreadResumeCallsCallbackWhenThreadIsStopped)
{
    // Initialize thread to be in stopped state.
    SetupKnownStoppedThread (NEW_THREAD_TID);

    // Request a resume.
    lldb::tid_t resumed_tid = 0;
    int resume_call_count = 0;

    m_coordinator.RequestThreadResume (NEW_THREAD_TID,
                                      [&](lldb::tid_t tid)
                                      {
                                          ++resume_call_count;
                                          resumed_tid = tid;
                                      },
                                      GetErrorFunction ());
    // Shouldn't be called yet.
    ASSERT_EQ (0, resume_call_count);

    // Process next event.  After that, the resume request call should have fired.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();
    ASSERT_EQ (1, resume_call_count);
    ASSERT_EQ (NEW_THREAD_TID, resumed_tid);
}

TEST_F (ThreadStateCoordinatorTest, RequestThreadResumeSignalsErrorOnAlreadyRunningThread)
{
    const lldb::tid_t TEST_TID = 1234;
    SetupKnownRunningThread (NEW_THREAD_TID);

    // Request a resume.
    lldb::tid_t resumed_tid = 0;
    int resume_call_count = 0;

    m_coordinator.RequestThreadResume (TEST_TID,
                                     [&](lldb::tid_t tid)
                                     {
                                         ++resume_call_count;
                                         resumed_tid = tid;
                                     },
                                     GetErrorFunction ());

    // Shouldn't be called yet.
    ASSERT_EQ (0, resume_call_count);

    // Process next event.  Should be an error.
    ASSERT_PROCESS_NEXT_EVENT_FAILS ();

    // The resume request should not have gone off because we think it is already running.
    ASSERT_EQ (0, resume_call_count);
}

TEST_F (ThreadStateCoordinatorTest, ResumedThreadAlreadyMarkedDoesNotHoldUpPendingStopNotification)
{
    // We're going to test this scenario:
    //   * Deferred notification waiting on two threads, A and B.  A and B currently running.
    //   * Thread A stops.
    //   * Thread A resumes.
    //   * Thread B stops.
    //
    //   Here we could have forced A to stop again (after the Thread A resumes) because we had a pending stop nofication awaiting
    //   all those threads to stop.  However, we are going to explicitly not try to restop A - somehow
    //   that seems wrong and possibly buggy since for that to happen, we would have intentionally called
    //   a resume after the stop.  Instead, we'll just log and indicate this looks suspicous.  We can revisit
    //   that decision after we see if/when that happens.
    const lldb::tid_t PENDING_TID_A = 2;
    const lldb::tid_t PENDING_TID_B = 89;

    SetupKnownStoppedThread (TRIGGERING_TID);
    SetupKnownRunningThread (PENDING_TID_A);
    SetupKnownRunningThread (PENDING_TID_B);

    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { PENDING_TID_A, PENDING_TID_B };

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    CallAfterThreadsStop (TRIGGERING_TID,
                          pending_stop_tids);

    // Neither trigger should have gone off yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());
    ASSERT_EQ (0, GetRequestedStopCount ());

    // Execute CallAfterThreadsStop.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Both TID A and TID B should have had stop requests made.
    ASSERT_EQ (2, GetRequestedStopCount ());
    ASSERT_EQ (true, DidRequestStopForTid (PENDING_TID_A));
    ASSERT_EQ (true, DidRequestStopForTid (PENDING_TID_B));

    // But we still shouldn't have the deferred signal call go off yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Report thread A stopped.
    NotifyThreadStop (PENDING_TID_A);
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Now report thread A is resuming.  Ensure the resume is called.
    bool resume_called = false;
    m_coordinator.RequestThreadResume (PENDING_TID_A,
                                     [&](lldb::tid_t tid) { resume_called = true; },
                                     GetErrorFunction ());
    ASSERT_EQ (false, resume_called);
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();
    ASSERT_EQ (true, resume_called);

    // Report thread B stopped.
    NotifyThreadStop (PENDING_TID_B);
    ASSERT_EQ (false, DidFireDeferredNotification ());
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // After notifying thread b stopped, we now have thread a resumed but thread b stopped.
    // However, since thread a had stopped, we now have had both requirements stopped at some point.
    // For now we'll expect this will fire the pending deferred stop notification.
    ASSERT_EQ (true, DidFireDeferredNotification ());
    ASSERT_EQ (TRIGGERING_TID, GetDeferredNotificationTID ());
}

TEST_F (ThreadStateCoordinatorTest, CallAfterRunningThreadsStopFiresWhenNoRunningThreads)
{
    // Let the coordinator know about our thread.
    SetupKnownStoppedThread (TRIGGERING_TID);

    // Notify we have a trigger that needs to be fired when all running threads have stopped.
    CallAfterRunningThreadsStop (TRIGGERING_TID);

    // Notification trigger shouldn't go off yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Process next event.  This will pick up the call after threads stop event.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Now the trigger should have fired, since there were no threads that needed to first stop.
    ASSERT_EQ (true, DidFireDeferredNotification ());
    ASSERT_EQ (TRIGGERING_TID, GetDeferredNotificationTID ());

    // And no stop requests should have been made.
    ASSERT_EQ (0, GetRequestedStopCount ());
}

TEST_F (ThreadStateCoordinatorTest, CallAfterRunningThreadsStopRequestsTwoPendingStops)
{
    // Let the coordinator know about our threads.
    SetupKnownStoppedThread (TRIGGERING_TID);
    SetupKnownRunningThread (PENDING_STOP_TID);
    SetupKnownRunningThread (PENDING_STOP_TID_02);

    // Notify we have a trigger that needs to be fired when all running threads have stopped.
    CallAfterRunningThreadsStop (TRIGGERING_TID);

    // Notification trigger shouldn't go off yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Process next event.  This will pick up the call after threads stop event.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // We should have two stop requests for the two threads currently running.
    ASSERT_EQ (2, GetRequestedStopCount ());
    ASSERT_EQ (true, DidRequestStopForTid (PENDING_STOP_TID));
    ASSERT_EQ (true, DidRequestStopForTid (PENDING_STOP_TID_02));

    // But the deferred stop notification should not have fired yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Now notify the two threads stopped.
    NotifyThreadStop (PENDING_STOP_TID);
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();
    ASSERT_EQ (false, DidFireDeferredNotification ());

    NotifyThreadStop (PENDING_STOP_TID_02);
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Now the trigger should have fired, since there were no threads that needed to first stop.
    ASSERT_EQ (true, DidFireDeferredNotification ());
    ASSERT_EQ (TRIGGERING_TID, GetDeferredNotificationTID ());
}

TEST_F (ThreadStateCoordinatorTest, CallAfterRunningThreadsStopRequestsStopTwoOtherThreadsOneRunning)
{
    // Let the coordinator know about our threads.  PENDING_STOP_TID_02 will already be stopped.
    SetupKnownStoppedThread (TRIGGERING_TID);
    SetupKnownRunningThread (PENDING_STOP_TID);
    SetupKnownStoppedThread (PENDING_STOP_TID_02);

    // Notify we have a trigger that needs to be fired when all running threads have stopped.
    CallAfterRunningThreadsStop (TRIGGERING_TID);

    // Notification trigger shouldn't go off yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Process next event.  This will pick up the call after threads stop event.
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // We should have two stop requests for the two threads currently running.
    ASSERT_EQ (1, GetRequestedStopCount ());
    ASSERT_EQ (true, DidRequestStopForTid (PENDING_STOP_TID));

    // But the deferred stop notification should not have fired yet.
    ASSERT_EQ (false, DidFireDeferredNotification ());

    // Now notify the two threads stopped.
    NotifyThreadStop (PENDING_STOP_TID);
    ASSERT_PROCESS_NEXT_EVENT_SUCCEEDS ();

    // Now the trigger should have fired, since there were no threads that needed to first stop.
    ASSERT_EQ (true, DidFireDeferredNotification ());
    ASSERT_EQ (TRIGGERING_TID, GetDeferredNotificationTID ());
}
