"""
Test lldb Python event APIs.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class EventAPITestCase(TestBase):

    mydir = os.path.join("python_api", "event")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_wait_for_event_with_dsym(self):
        """Exercise SBListener.WaitForEvent() API."""
        self.buildDsym()
        self.do_wait_for_event()

    @python_api_test
    def test_wait_for_event_with_dwarf(self):
        """Exercise SBListener.WaitForEvent() API."""
        self.buildDwarf()
        self.do_wait_for_event()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_add_listener_to_broadcaster_dsym(self):
        """Exercise some SBBroadcaster APIs."""
        self.buildDsym()
        self.do_add_listener_to_broadcaster()

    @python_api_test
    def test_add_listener_to_broadcaster_dwarf(self):
        """Exercise some SBBroadcaster APIs."""
        self.buildDwarf()
        self.do_add_listener_to_broadcaster()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line = line_number('main.c', '// Find the line number of function "c" here.')

    def do_wait_for_event(self):
        """Get the listener associated with the debugger and exercise WaitForEvent API."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        #print "breakpoint:", breakpoint
        self.assertTrue(breakpoint.IsValid() and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        self.process = target.LaunchProcess([], [], os.ctermid(), 0, False)

        self.process = target.GetProcess()
        self.assertTrue(self.process.IsValid(), PROCESS_IS_VALID)

        # Get a handle on the process's broadcaster.
        broadcaster = self.process.GetBroadcaster()
        self.assertTrue(broadcaster.IsValid(), "Process with valid broadcaster")

        # Create an empty event object.
        event = lldb.SBEvent()
        self.assertFalse(event.IsValid(), "Event should not be valid initially")

        # Get the debugger listener.
        listener = self.dbg.GetListener()

        # Create MyListeningThread to wait for any kind of event.
        import threading
        class MyListeningThread(threading.Thread):
            def run(self):
                count = 0
                # Let's only try at most 3 times to retrieve any kind of event.
                while not count > 3:
                    if listener.WaitForEvent(5, event):
                        #print "Got a valid event:", event
                        return
                    count = count + 1
                    print "Timeout: listener.WaitForEvent"

                return

        # Use Python API to kill the process.  The listening thread should be
        # able to receive a state changed event.
        self.process.Kill()

        # Let's start the listening thread to retrieve the event.
        my_thread = MyListeningThread()
        my_thread.start()

        # Wait until the 'MyListeningThread' terminates.
        my_thread.join()

        self.assertTrue(event.IsValid(),
                        "My listening thread successfully received an event")

    def do_add_listener_to_broadcaster(self):
        """Get the broadcaster associated with the process and wait for broadcaster events."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        #print "breakpoint:", breakpoint
        self.assertTrue(breakpoint.IsValid() and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        self.process = target.LaunchProcess([], [], os.ctermid(), 0, False)

        self.process = target.GetProcess()
        self.assertTrue(self.process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        # Get a handle on the process's broadcaster.
        broadcaster = self.process.GetBroadcaster()
        self.assertTrue(broadcaster.IsValid(), "Process with valid broadcaster")

        # Create an empty event object.
        event = lldb.SBEvent()
        self.assertFalse(event.IsValid(), "Event should not be valid initially")

        # Create a listener object and register with the broadcaster.
        listener = lldb.SBListener("TestEvents.listener")
        rc = broadcaster.AddListener(listener, lldb.SBProcess.eBroadcastBitStateChanged)
        self.assertTrue(rc, "AddListener successfully retruns")

        # The finite state machine for our custom listening thread, with an
        # initail state of 0, which means a "running" event is expected.
        # It changes to 1 after "running" is received.
        # It cahnges to 2 after "stopped" is received.
        # 2 will be our final state and the test is complete.
        self.state = 0 

        # Create MyListeningThread to wait for state changed events.
        # By design, a "running" event is expected following by a "stopped" event.
        import threading
        class MyListeningThread(threading.Thread):
            def run(self):
                #print "Running MyListeningThread:", self

                # Regular expression pattern for the event description.
                pattern = re.compile("data = {.*, state = (.*)}$")

                # Let's only try at most 6 times to retrieve our events.
                count = 0
                while True:
                    if listener.WaitForEventForBroadcasterWithType(5,
                                                                   broadcaster,
                                                                   lldb.SBProcess.eBroadcastBitStateChanged,
                                                                   event):
                        stream = lldb.SBStream()
                        event.GetDescription(stream)
                        description = stream.GetData()
                        #print "Event description:", description
                        match = pattern.search(description)
                        if not match:
                            break;
                        if self.context.state == 0 and match.group(1) == 'running':
                            self.context.state = 1
                            continue
                        elif self.context.state == 1 and match.group(1) == 'stopped':
                            # Whoopee, both events have been received!
                            self.context.state = 2
                            break
                        else:
                            break
                    print "Timeout: listener.WaitForEvent"
                    count = count + 1
                    if count > 6:
                        break

                return

        # Use Python API to continue the process.  The listening thread should be
        # able to receive the state changed events.
        self.process.Continue()

        # Start the listening thread to receive the "running" followed by the
        # "stopped" events.
        my_thread = MyListeningThread()
        # Supply the enclosing context so that our listening thread can access
        # the 'state' variable.
        my_thread.context = self
        my_thread.start()

        # Wait until the 'MyListeningThread' terminates.
        my_thread.join()

        # We are no longer interested in receiving state changed events.
        # Remove our custom listener before the inferior is killed.
        broadcaster.RemoveListener(listener, lldb.SBProcess.eBroadcastBitStateChanged)

        # The final judgement. :-)
        self.assertTrue(self.state == 2,
                        "Both expected state changed events received")

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
