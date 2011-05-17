"""
Test utility functions for the frame object.
"""

import os
import unittest2
import lldb
from lldbtest import *

class FrameUtilsTestCase(TestBase):

    mydir = "python_api/lldbutil/frame"

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c',
                                "// Find the line number here.")

    def test_frame_utils(self):
        """Test utility functions for the frame object."""
        self.buildDefault(dictionary={'C_SOURCES': 'main.c'})
        self.frame_utils()

    def frame_utils(self):
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.c", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        self.process = target.LaunchSimple(None, None, os.getcwd())

        if not self.process:
            self.fail("SBTarget.LaunchProcess() failed")
        self.assertTrue(self.process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        import lldbutil
        thread = lldbutil.get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        frame0 = thread.GetFrameAtIndex(0)
        frame1 = thread.GetFrameAtIndex(1)
        parent = lldbutil.get_parent_frame(frame0)
        self.assertTrue(parent and parent.GetFrameID() == frame1.GetFrameID())
        frame0_args = lldbutil.get_args_as_string(frame0)
        parent_args = lldbutil.get_args_as_string(parent)
        self.assertTrue(frame0_args and parent_args)
        if self.TraceOn():
            lldbutil.print_stacktrace(thread)
            print "Current frame: %s" % frame0_args
            print "Parent frame: %s" % parent_args


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
