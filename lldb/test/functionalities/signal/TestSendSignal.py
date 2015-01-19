"""Test that lldb command 'process signal SIGUSR1' to send a signal to the inferior works."""

import os, time, signal
import unittest2
import lldb
from lldbtest import *
import lldbutil

class SendSignalTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test that lldb command 'process signal SIGUSR1' sends a signal to the inferior process."""
        self.buildDsym()
        self.send_signal()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Test that lldb command 'process signal SIGUSR1' sends a signal to the inferior process."""
        self.buildDwarf()
        self.send_signal()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', 'Put breakpoint here')

    def send_signal(self):
        """Test that lldb command 'process signal SIGUSR1' sends a signal to the inferior process."""

        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByLocation ('main.c', self.line)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Get the breakpoint location from breakpoint after we verified that,
        # indeed, it has one location.
        location = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location and
                        location.IsEnabled(),
                        VALID_BREAKPOINT_LOCATION)

        # Now launch the process, no arguments & do not stop at entry point.
        launch_info = lldb.SBLaunchInfo([exe])
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        
        process_listener = lldb.SBListener("signal_test_listener")
        launch_info.SetListener(process_listener)
        error = lldb.SBError()
        process = target.Launch (launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        self.runCmd("process handle -n False -p True -s True SIGUSR1")

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "We hit the first breakpoint.")

        # After resuming the process, send it a SIGUSR1 signal.

        self.setAsync(True)

        broadcaster = process.GetBroadcaster()

        self.assertTrue(process_listener.IsValid(), "Got a good process listener")

        # Disable our breakpoint, we don't want to hit it anymore...
        breakpoint.SetEnabled(False)

        # Now continue:
        process.Continue()

        event = lldb.SBEvent()
        got_event = process_listener.WaitForEventForBroadcasterWithType(5, broadcaster, lldb.SBProcess.eBroadcastBitStateChanged, event)
        event_type = lldb.SBProcess.GetStateFromEvent(event)
        self.assertTrue (got_event, "Got an event")
        self.assertTrue (event_type == lldb.eStateRunning, "It was the running event.")
        
        # Now signal the process, and make sure it stops:
        process.Signal(signal.SIGUSR1)

        got_event = process_listener.WaitForEventForBroadcasterWithType(5, broadcaster, lldb.SBProcess.eBroadcastBitStateChanged, event)

        event_type = lldb.SBProcess.GetStateFromEvent(event)
        self.assertTrue (got_event, "Got an event")
        self.assertTrue (event_type == lldb.eStateStopped, "It was the stopped event.")
        
        # Now make sure the thread was stopped with a SIGUSR1:
        threads = lldbutil.get_stopped_threads (process, lldb.eStopReasonSignal)
        self.assertTrue (len(threads) == 1, "One thread stopped for a signal.")
        thread = threads[0]

        self.assertTrue (thread.GetStopReasonDataCount() >= 1, "There was data in the event.")
        self.assertTrue (thread.GetStopReasonDataAtIndex(0) == signal.SIGUSR1, "The stop signal was SIGUSR1")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
