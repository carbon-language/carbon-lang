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
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break by function name a_function (not yet loaded).
        self.expect("breakpoint set -n a_function", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: name = 'a_function', locations = 0 (pending)")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint and at a_function.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is Stopped',
                       'a_function',
                       'a.c:14',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

#         # Issue the 'contnue' command.  We should stop agaian at a_function.
#         # The stop reason of the thread should be breakpoint and at a_function.
#         self.runCmd("continue")
#         self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
#             substrs = ['state is Stopped',
#                        'a_function',
#                        'a.c:14',
#                        'stop reason = breakpoint'])
#
#         # The breakpoint should have a hit count of 2.
#         self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
#             substrs = [' resolved, hit count = 2'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
