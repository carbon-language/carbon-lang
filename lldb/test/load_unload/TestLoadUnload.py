"""
Test that breakpoint by symbol name works correctly dlopen'ing a dynamic lib.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestLoadUnload(TestBase):

    mydir = "load_unload"

    def test_load_unload(self):
        """Test breakpoint by name works correctly with dlopen'ing."""
        res = self.res
        exe = os.path.join(os.getcwd(), "a.out")
        self.ci.HandleCommand("file " + exe, res)
        self.assertTrue(res.Succeeded(), CURRENT_EXECUTABLE_SET)

        # Break by function name a_function (not yet loaded).
        self.ci.HandleCommand("breakpoint set -n a_function", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 1: name = 'a_function', locations = 0 "
            "(pending)"
            ),
                        BREAKPOINT_CREATED)

        self.ci.HandleCommand("run", res)
        self.runStarted = True
        self.assertTrue(res.Succeeded(), RUN_STOPPED)

        # The stop reason of the thread should be breakpoint and at a_function.
        self.ci.HandleCommand("thread list", res)
        output = res.GetOutput()
        self.assertTrue(res.Succeeded(), CMD_MSG('thread list'))
        self.assertTrue(output.find('state is Stopped') > 0 and
                        output.find('a_function') > 0 and
                        output.find('a.c:14') > 0 and
                        output.find('stop reason = breakpoint') > 0,
                        STOPPED_DUE_TO_BREAKPOINT)

        # The breakpoint should have a hit count of 1.
        self.ci.HandleCommand("breakpoint list", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find(' resolved, hit count = 1') > 0,
                        BREAKPOINT_HIT_ONCE)

#         # We should stop agaian at a_function.
#         # The stop reason of the thread should be breakpoint and at a_function.
#         self.ci.HandleCommand("thread list", res)
#         output = res.GetOutput()
#         self.assertTrue(res.Succeeded())
#         self.assertTrue(output.find('state is Stopped') > 0 and
#                         output.find('a_function') > 0 and
#                         output.find('a.c:14') > 0 and
#                         output.find('stop reason = breakpoint') > 0)

#         # The breakpoint should have a hit count of 2.
#         self.ci.HandleCommand("breakpoint list", res)
#         self.assertTrue(res.Succeeded())
#         self.assertTrue(res.GetOutput().find(' resolved, hit count = 2') > 0)

#         self.ci.HandleCommand("continue", res)
#         self.assertTrue(res.Succeeded())


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
