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
        """Exercise SBListener.WaitForEvent() APIs."""
        self.buildDsym()
        self.do_wait_for_event()

    @python_api_test
    def test_wait_for_event_with_dwarf(self):
        """Exercise SBListener.WaitForEvent() APIs."""
        self.buildDwarf()
        self.do_wait_for_event()

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
        self.process = target.LaunchProcess([''], [''], os.ctermid(), 0, False)

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
                print "Running MyListeningThread:", self
                count = 0
                # Let's only try at most 3 times to retrieve any kind of event.
                while not count > 3:
                    if listener.WaitForEvent(5, event):
                        print "Got a valid event:", event
                        print "Event type:", event.GetType()
                        print "Event broadcaster:", event.GetBroadcaster().GetName()
                        return
                    count = count + 1
                    print "Timeout: listener.WaitForEvent"

                return

        # Let's start the listening thread before we launch the inferior process.
        my_thread = MyListeningThread()
        my_thread.start()

        # Set the debugger to be in asynchronous mode since our listening thread
        # is waiting for events to come.
        self.dbg.SetAsync(True)

        # Use Python API to kill the process.  The listening thread should be
        # able to receive a state changed event.
        self.process.Kill()

        # Wait until the 'MyListeningThread' terminates.
        my_thread.join()

        # Restore the original synchronous mode.
        self.dbg.SetAsync(False)

        self.assertTrue(event.IsValid())

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
