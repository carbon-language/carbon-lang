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

    ASSERT_EQ(coordinator.ProcessNextEvent (), false);
}

TEST(ThreadStateCoordinatorTest, CallAfterThreadsStopFiresWhenNoPendingStops)
{
    ThreadStateCoordinator coordinator(NOPLogger);

    const lldb::tid_t TRIGGERING_TID = 4105;
    bool call_after_fired = false;

    // Notify we have a trigger that needs to be fired when all threads in the wait tid set have stopped.
    coordinator.CallAfterThreadsStop (TRIGGERING_TID,
                                      EMPTY_THREAD_ID_SET,
                                      [](lldb::tid_t tid) {},
                                      [&](lldb::tid_t tid) { call_after_fired = true; });

    // Notification trigger shouldn't go off yet.
    ASSERT_EQ (call_after_fired, false);

    // Process next event.  This will pick up the call after threads stop event.
    ASSERT_EQ(coordinator.ProcessNextEvent (), true);

    // Now the trigger should have fired.
    ASSERT_EQ(call_after_fired, true);
}
