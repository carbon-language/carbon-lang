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
