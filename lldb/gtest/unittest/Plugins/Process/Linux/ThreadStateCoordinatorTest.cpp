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
}

TEST(ThreadStateCoordinatorTest, StopCoordinatorWorksNoPriorEvents)
{
    ThreadStateCoordinator coordinator(NOPLogger);

    coordinator.StopCoordinator ();

    ASSERT_EQ(false, coordinator.ProcessNextEvent ());
}

TEST(ThreadStateCoordinatorTest, CallAfterThreadsStopFiresWhenNoPendingStops)
{
    ThreadStateCoordinator coordinator(NOPLogger);

    const lldb::tid_t TRIGGERING_TID = 4105;
    bool call_after_fired = false;
    lldb::tid_t reported_firing_tid = 0;

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    coordinator.CallAfterThreadsStop (TRIGGERING_TID,
                                      EMPTY_THREAD_ID_SET,
                                      [](lldb::tid_t tid) {},
                                      [&](lldb::tid_t tid) {
                                          call_after_fired = true;
                                          reported_firing_tid = tid;
                                      });

    // Notification trigger shouldn't go off yet.
    ASSERT_EQ (false, call_after_fired);

    // Process next event.  This will pick up the call after threads stop event.
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // Now the trigger should have fired, since there were no threads that needed to first stop.
    ASSERT_EQ (true, call_after_fired);

    // And the firing tid should be the one we indicated.
    ASSERT_EQ (TRIGGERING_TID, reported_firing_tid);
}

TEST(ThreadStateCoordinatorTest, CallAfterThreadsStopFiresWhenOnePendingStop)
{
    ThreadStateCoordinator coordinator(NOPLogger);

    const lldb::tid_t TRIGGERING_TID = 4105;
    const lldb::tid_t PENDING_STOP_TID = 3;

    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { PENDING_STOP_TID };

    bool call_after_fired = false;
    lldb::tid_t reported_firing_tid = 0;

    bool request_thread_stop_called = false;
    lldb::tid_t request_thread_stop_tid = 0;

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    coordinator.CallAfterThreadsStop (TRIGGERING_TID,
                                      pending_stop_tids,
                                      [&](lldb::tid_t tid) {
                                          request_thread_stop_called = true;
                                          request_thread_stop_tid = tid;

                                      },
                                      [&](lldb::tid_t tid) {
                                          call_after_fired = true;
                                          reported_firing_tid = tid;
                                      });

    // Neither trigger should have gone off yet.
    ASSERT_EQ (false, call_after_fired);
    ASSERT_EQ (false, request_thread_stop_called);

    // Process next event.
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // Now the request thread stop should have been called for the pending stop.
    ASSERT_EQ (true, request_thread_stop_called);
    ASSERT_EQ (PENDING_STOP_TID, request_thread_stop_tid);

    // But we still shouldn't have the deferred signal call go off yet.  Need to wait for the stop to be reported.
    ASSERT_EQ (false, call_after_fired);

    // Now report the that the pending stop occurred.
    coordinator.NotifyThreadStop (PENDING_STOP_TID);

    // Shouldn't take effect until after next processing step.
    ASSERT_EQ (false, call_after_fired);

    // Process next event.
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // Deferred signal notification should have fired now.
    ASSERT_EQ (true, call_after_fired);
    ASSERT_EQ (TRIGGERING_TID, reported_firing_tid);
}

TEST(ThreadStateCoordinatorTest, CallAfterThreadsStopFiresWhenTwoPendingStops)
{
    ThreadStateCoordinator coordinator(NOPLogger);

    const lldb::tid_t TRIGGERING_TID = 4105;
    const lldb::tid_t PENDING_STOP_TID = 3;
    const lldb::tid_t PENDING_STOP_TID_02 = 29016;

    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { PENDING_STOP_TID, PENDING_STOP_TID_02 };

    bool call_after_fired = false;
    lldb::tid_t reported_firing_tid = 0;

    int request_thread_stop_calls = 0;
    ThreadStateCoordinator::ThreadIDSet request_thread_stop_tids;

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    coordinator.CallAfterThreadsStop (TRIGGERING_TID,
                                      pending_stop_tids,
                                      [&](lldb::tid_t tid) {
                                          ++request_thread_stop_calls;
                                          request_thread_stop_tids.insert (tid);

                                      },
                                      [&](lldb::tid_t tid) {
                                          call_after_fired = true;
                                          reported_firing_tid = tid;
                                      });

    // Neither trigger should have gone off yet.
    ASSERT_EQ (false, call_after_fired);
    ASSERT_EQ (0, request_thread_stop_calls);

    // Process next event.
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // Now the request thread stops should have been called for the pending stop tids.
    ASSERT_EQ (2, request_thread_stop_calls);
    ASSERT_EQ (1, request_thread_stop_tids.count (PENDING_STOP_TID));
    ASSERT_EQ (1, request_thread_stop_tids.count (PENDING_STOP_TID_02));

    // But we still shouldn't have the deferred signal call go off yet.  Need to wait for the stop to be reported.
    ASSERT_EQ (false, call_after_fired);

    // Report the that the first pending stop occurred.
    coordinator.NotifyThreadStop (PENDING_STOP_TID);
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // Shouldn't take effect until after both pending threads are notified.
    ASSERT_EQ (false, call_after_fired);

    // Report the that the first pending stop occurred.
    coordinator.NotifyThreadStop (PENDING_STOP_TID_02);
    ASSERT_EQ (false, call_after_fired);
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // Deferred signal notification should have fired now.
    ASSERT_EQ (true, call_after_fired);
    ASSERT_EQ (TRIGGERING_TID, reported_firing_tid);
}

TEST(ThreadStateCoordinatorTest, CallAfterThreadsStopFiresWhenPendingAlreadyStopped)
{
    ThreadStateCoordinator coordinator(NOPLogger);

    const lldb::tid_t TRIGGERING_TID = 4105;
    const lldb::tid_t PENDING_STOP_TID = 3;

    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { PENDING_STOP_TID };

    // Tell coordinator the pending stop tid is already stopped.
    coordinator.NotifyThreadStop (PENDING_STOP_TID);
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // Now fire the deferred thread stop notification, indicating that the pending thread
    // must be stopped before we notify.
    bool call_after_fired = false;
    lldb::tid_t reported_firing_tid = 0;

    bool request_thread_stop_called = false;
    lldb::tid_t request_thread_stop_tid = 0;

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    coordinator.CallAfterThreadsStop (TRIGGERING_TID,
                                      pending_stop_tids,
                                      [&](lldb::tid_t tid) {
                                          request_thread_stop_called = true;
                                          request_thread_stop_tid = tid;

                                      },
                                      [&](lldb::tid_t tid) {
                                          call_after_fired = true;
                                          reported_firing_tid = tid;
                                      });

    // Neither trigger should have gone off yet.
    ASSERT_EQ (false, call_after_fired);
    ASSERT_EQ (false, request_thread_stop_called);

    // Process next event.
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // The pending stop should *not* fire because the coordinator knows it has already stopped.
    ASSERT_EQ (false, request_thread_stop_called);

    // The deferred signal notification should have fired since all requirements were met.
    ASSERT_EQ (true, call_after_fired);
    ASSERT_EQ (TRIGGERING_TID, reported_firing_tid);
}

TEST(ThreadStateCoordinatorTest, CallAfterThreadsStopFiresWhenTwoPendingOneAlreadyStopped)
{
    ThreadStateCoordinator coordinator(NOPLogger);

    const lldb::tid_t TRIGGERING_TID = 4105;
    const lldb::tid_t PENDING_STOP_TID = 3;
    const lldb::tid_t PENDING_STOP_TID_02 = 29016;

    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { PENDING_STOP_TID, PENDING_STOP_TID_02 };

    // Tell coordinator the pending stop tid is already stopped.
    coordinator.NotifyThreadStop (PENDING_STOP_TID);
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // Now fire the deferred thread stop notification, indicating that the pending thread
    // must be stopped before we notify.
    bool call_after_fired = false;
    lldb::tid_t reported_firing_tid = 0;

    int request_thread_stop_calls = 0;
    lldb::tid_t request_thread_stop_tid = 0;

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    coordinator.CallAfterThreadsStop (TRIGGERING_TID,
                                      pending_stop_tids,
                                      [&](lldb::tid_t tid) {
                                          ++request_thread_stop_calls;
                                          request_thread_stop_tid = tid;

                                      },
                                      [&](lldb::tid_t tid) {
                                          call_after_fired = true;
                                          reported_firing_tid = tid;
                                      });

    // Neither trigger should have gone off yet.
    ASSERT_EQ (false, call_after_fired);
    ASSERT_EQ (0, request_thread_stop_calls);

    // Process next event.
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // The pending stop should only fire for one of the threads, the one that wasn't already stopped.
    ASSERT_EQ (1, request_thread_stop_calls);
    ASSERT_EQ (PENDING_STOP_TID_02, request_thread_stop_tid);

    // The deferred signal notification should not yet have fired since all pending thread stops have not yet occurred.
    ASSERT_EQ (false, call_after_fired);

    // Notify final thread has stopped.
    coordinator.NotifyThreadStop (PENDING_STOP_TID_02);
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // The deferred signal notification should have fired since all requirements were met.
    ASSERT_EQ (true, call_after_fired);
    ASSERT_EQ (TRIGGERING_TID, reported_firing_tid);
}

TEST(ThreadStateCoordinatorTest, CallAfterThreadsStopFiresWhenOnePendingThreadDies)
{
    ThreadStateCoordinator coordinator(NOPLogger);

    const lldb::tid_t TRIGGERING_TID = 4105;
    const lldb::tid_t PENDING_STOP_TID = 3;

    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { PENDING_STOP_TID };

    bool call_after_fired = false;
    lldb::tid_t reported_firing_tid = 0;

    bool request_thread_stop_called = false;
    lldb::tid_t request_thread_stop_tid = 0;

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    coordinator.CallAfterThreadsStop (TRIGGERING_TID,
                                      pending_stop_tids,
                                      [&](lldb::tid_t tid) {
                                          request_thread_stop_called = true;
                                          request_thread_stop_tid = tid;

                                      },
                                      [&](lldb::tid_t tid) {
                                          call_after_fired = true;
                                          reported_firing_tid = tid;
                                      });

    // Neither trigger should have gone off yet.
    ASSERT_EQ (false, call_after_fired);
    ASSERT_EQ (false, request_thread_stop_called);

    // Process next event.
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // Now the request thread stop should have been called for the pending stop.
    ASSERT_EQ (true, request_thread_stop_called);
    ASSERT_EQ (PENDING_STOP_TID, request_thread_stop_tid);

    // But we still shouldn't have the deferred signal call go off yet.  Need to wait for the death to be reported.
    ASSERT_EQ (false, call_after_fired);

    // Now report the that the thread with pending stop dies.
    coordinator.NotifyThreadDeath (PENDING_STOP_TID);

    // Shouldn't take effect until after next processing step.
    ASSERT_EQ (false, call_after_fired);

    // Process next event.
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // Deferred signal notification should have fired now.
    ASSERT_EQ (true, call_after_fired);
    ASSERT_EQ (TRIGGERING_TID, reported_firing_tid);
}

TEST(ThreadStateCoordinatorTest, ExistingPendingNotificationRequiresStopFromNewThread)
{
    ThreadStateCoordinator coordinator(NOPLogger);

    const lldb::tid_t TRIGGERING_TID = 4105;
    const lldb::tid_t PENDING_STOP_TID = 3;

    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { PENDING_STOP_TID };

    bool call_after_fired = false;
    lldb::tid_t reported_firing_tid = 0;

    int request_thread_stop_calls = 0;
    ThreadStateCoordinator::ThreadIDSet request_thread_stop_tids;

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    coordinator.CallAfterThreadsStop (TRIGGERING_TID,
                                      pending_stop_tids,
                                      [&](lldb::tid_t tid) {
                                          ++request_thread_stop_calls;
                                          request_thread_stop_tids.insert (tid);

                                      },
                                      [&](lldb::tid_t tid) {
                                          call_after_fired = true;
                                          reported_firing_tid = tid;
                                      });

    // Neither trigger should have gone off yet.
    ASSERT_EQ (false, call_after_fired);
    ASSERT_EQ (0, request_thread_stop_calls);

    // Process next event.
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // Now the request thread stop should have been called for the pending stop.
    ASSERT_EQ (1, request_thread_stop_calls);
    ASSERT_EQ (true, request_thread_stop_tids.count (PENDING_STOP_TID));

    // But we still shouldn't have the deferred signal call go off yet.
    ASSERT_EQ (false, call_after_fired);

    // Indicate a new thread has just been created.
    const lldb::tid_t NEW_THREAD_TID = 1234;

    coordinator.NotifyThreadCreate (NEW_THREAD_TID);
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // We should have just received a stop request for the new thread id.
    ASSERT_EQ (2, request_thread_stop_calls);
    ASSERT_EQ (true, request_thread_stop_tids.count (NEW_THREAD_TID));

    // Now report the original pending tid stopped.  This should no longer
    // trigger the pending notification because we should now require the
    // new thread to stop too.
    coordinator.NotifyThreadStop (PENDING_STOP_TID);
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());
    ASSERT_EQ (false, call_after_fired);

    // Now notify the new thread stopped.
    coordinator.NotifyThreadStop (NEW_THREAD_TID);
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // Deferred signal notification should have fired now.
    ASSERT_EQ (true, call_after_fired);
    ASSERT_EQ (TRIGGERING_TID, reported_firing_tid);
}

TEST(ThreadStateCoordinatorTest, DeferredNotificationRemovedByResetForExec)
{
    ThreadStateCoordinator coordinator(NOPLogger);

    const lldb::tid_t TRIGGERING_TID = 4105;
    bool call_after_fired = false;
    lldb::tid_t reported_firing_tid = 0;

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    coordinator.CallAfterThreadsStop (TRIGGERING_TID,
                                      EMPTY_THREAD_ID_SET,
                                      [](lldb::tid_t tid) {},
                                      [&](lldb::tid_t tid) {
                                          call_after_fired = true;
                                          reported_firing_tid = tid;
                                      });

    // Notification trigger shouldn't go off yet.
    ASSERT_EQ (false, call_after_fired);

    // Now indicate an exec occurred, which will invalidate all state about the process and threads.
    coordinator.ResetForExec ();

    // Verify the deferred stop notification does *not* fire with the next
    // process.  It will handle the reset and not the deferred signaling, which
    // should now be removed.
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());
    ASSERT_EQ (false, call_after_fired);
}


TEST(ThreadStateCoordinatorTest, RequestThreadResumeCallsCallbackWhenThreadIsStopped)
{
    ThreadStateCoordinator coordinator(NOPLogger);

    // Initialize thread to be in stopped state.
    const lldb::tid_t TEST_TID = 1234;

    coordinator.NotifyThreadStop (TEST_TID);
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // Request a resume.
    lldb::tid_t resumed_tid = 0;
    int resume_call_count = 0;

    coordinator.RequestThreadResume (TEST_TID,
                                     [&](lldb::tid_t tid)
                                     {
                                         ++resume_call_count;
                                         resumed_tid = tid;
                                     });

    // Shouldn't be called yet.
    ASSERT_EQ (0, resume_call_count);

    // Process next event.  After that, the resume request call should have fired.
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());
    ASSERT_EQ (1, resume_call_count);
    ASSERT_EQ (TEST_TID, resumed_tid);
}

TEST(ThreadStateCoordinatorTest, RequestThreadResumeIgnoresCallbackWhenThreadIsRunning)
{
    ThreadStateCoordinator coordinator(StdoutLogger);

    // This thread will be assumed running (i.e. unknown, assumed running until marked stopped.)
    const lldb::tid_t TEST_TID = 1234;

    // Request a resume.
    lldb::tid_t resumed_tid = 0;
    int resume_call_count = 0;

    coordinator.RequestThreadResume (TEST_TID,
                                     [&](lldb::tid_t tid)
                                     {
                                         ++resume_call_count;
                                         resumed_tid = tid;
                                     });

    // Shouldn't be called yet.
    ASSERT_EQ (0, resume_call_count);

    // Process next event.
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // The resume request should not have gone off because we think it is already running.
    ASSERT_EQ (0, resume_call_count);
}

TEST(ThreadStateCoordinatorTest, ResumedThreadAlreadyMarkedDoesNotHoldUpPendingStopNotification)
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

    ThreadStateCoordinator coordinator(StdoutLogger);

    const lldb::tid_t TRIGGERING_TID = 4105;
    const lldb::tid_t PENDING_TID_A = 2;
    const lldb::tid_t PENDING_TID_B = 89;

    ThreadStateCoordinator::ThreadIDSet pending_stop_tids { PENDING_TID_A, PENDING_TID_B };

    bool call_after_fired = false;
    lldb::tid_t reported_firing_tid = 0;

    int request_thread_stop_calls = 0;
    ThreadStateCoordinator::ThreadIDSet request_thread_stop_tids;

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    coordinator.CallAfterThreadsStop (TRIGGERING_TID,
                                      pending_stop_tids,
                                      [&](lldb::tid_t tid) {
                                          ++request_thread_stop_calls;
                                          request_thread_stop_tids.insert (tid);

                                      },
                                      [&](lldb::tid_t tid) {
                                          call_after_fired = true;
                                          reported_firing_tid = tid;
                                      });

    // Neither trigger should have gone off yet.
    ASSERT_EQ (false, call_after_fired);
    ASSERT_EQ (0, request_thread_stop_calls);

    // Execute CallAfterThreadsStop.
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // Both TID A and TID B should have had stop requests made.
    ASSERT_EQ (2, request_thread_stop_calls);
    ASSERT_EQ (1, request_thread_stop_tids.count (PENDING_TID_A));

    // But we still shouldn't have the deferred signal call go off yet.
    ASSERT_EQ (false, call_after_fired);

    // Report thread A stopped.
    coordinator.NotifyThreadStop (PENDING_TID_A);
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());
    ASSERT_EQ (false, call_after_fired);

    // Now report thread A is resuming.  Ensure the resume is called.
    bool resume_called = false;
    coordinator.RequestThreadResume (PENDING_TID_A, [&](lldb::tid_t tid) { resume_called = true; } );
    ASSERT_EQ (false, resume_called);
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());
    ASSERT_EQ (true, resume_called);

    // Report thread B stopped.
    coordinator.NotifyThreadStop (PENDING_TID_B);
    ASSERT_EQ (false, call_after_fired);
    ASSERT_EQ (true, coordinator.ProcessNextEvent ());

    // After notifying thread b stopped, we now have thread a resumed but thread b stopped.
    // However, since thread a had stopped, we now have had both requirements stopped at some point.
    // For now we'll expect this will fire the pending deferred stop notification.
    ASSERT_EQ (true, call_after_fired);
}
