"""
Test that break on a struct declaration has no effect.

Instead, the first executable statement is set as the breakpoint.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestStructTypes(TestBase):

    mydir = "struct_types"

    def test_struct_types(self):
        """Test that break on a struct declaration has no effect."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break on the ctor function of class C.
        self.expect("breakpoint set -f main.c -l 14", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = 14, locations = 1")

        self.runCmd("run", RUN_STOPPED)

        # We should be stopped on the first executable statement within the
        # function where the original breakpoint was attempted.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['main.c:20',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
