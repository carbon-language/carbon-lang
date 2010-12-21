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
    def test_with_dsym(self):
        """Exercise SBEvent APIs."""
        self.buildDsym()
        self.do_events()

    @python_api_test
    def test_with_dwarf(self):
        """Exercise SBEvent APIs."""
        self.buildDwarf()
        self.do_events()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line = line_number('main.c', '// Find the line number of function "c" here.')

    def do_events(self):
        """Get the listener associated with the debugger and exercise some event APIs."""
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

        # Use Python API to continue the process.  The listening thread should be
        # able to receive a state changed event.
        self.process.Continue()

        my_thread.join()
        self.assertTrue(event.IsValid())

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
