"""
Test that breakpoint works correctly in the presence of dead-code stripping.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestDeadStrip(TestBase):

    mydir = "dead-strip"

    def test_dead_strip(self):
        """Test breakpoint works correctly with dead-code stripping."""
        res = self.res
        exe = os.path.join(os.getcwd(), "a.out")
        self.ci.HandleCommand("file " + exe, res)
        self.assertTrue(res.Succeeded(), CURRENT_EXECUTABLE_SET)

        # Break by function name f1 (live code).
        self.ci.HandleCommand("breakpoint set -s a.out -n f1", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 1: name = 'f1', module = a.out, locations = 1"
            ),
                        BREAKPOINT_CREATED)

        # Break by function name f2 (dead code).
        self.ci.HandleCommand("breakpoint set -s a.out -n f2", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 2: name = 'f2', module = a.out, locations = 0 "
            "(pending)"),
                        BREAKPOINT_PENDING_CREATED)

        # Break by function name f3 (live code).
        self.ci.HandleCommand("breakpoint set -s a.out -n f3", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 3: name = 'f3', module = a.out, locations = 1"
            ),
                        BREAKPOINT_CREATED)

        self.ci.HandleCommand("run", res)
        self.runStarted = True
        self.assertTrue(res.Succeeded(), RUN_STOPPED)

        # The stop reason of the thread should be breakpoint (breakpoint #1).
        self.ci.HandleCommand("thread list", res)
        output = res.GetOutput()
        self.assertTrue(res.Succeeded())
        self.assertTrue(output.find('state is Stopped') > 0 and
                        output.find('main.c:20') > 0 and
                        output.find('where = a.out`f1') > 0 and
                        output.find('stop reason = breakpoint') > 0,
                        STOPPED_DUE_TO_BREAKPOINT)

        # The breakpoint should have a hit count of 1.
        self.ci.HandleCommand("breakpoint list 1", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find(' resolved, hit count = 1') > 0,
                        BREAKPOINT_HIT_ONCE)

        self.ci.HandleCommand("continue", res)
        self.assertTrue(res.Succeeded())

        # The stop reason of the thread should be breakpoint (breakpoint #3).
        self.ci.HandleCommand("thread list", res)
        output = res.GetOutput()
        self.assertTrue(res.Succeeded())
        self.assertTrue(output.find('state is Stopped') > 0 and
                        output.find('main.c:40') > 0 and
                        output.find('where = a.out`f3') > 0 and
                        output.find('stop reason = breakpoint') > 0,
                        STOPPED_DUE_TO_BREAKPOINT)

        # The breakpoint should have a hit count of 1.
        self.ci.HandleCommand("breakpoint list 3", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find(' resolved, hit count = 1') > 0,
                        BREAKPOINT_HIT_ONCE)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
