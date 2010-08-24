"""Show bitfields and check that they display correctly."""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestBitfields(TestBase):

    mydir = "bitfields"

    @unittest2.expectedFailure
    def test_global_variables(self):
        """Test 'variable list ...' and check for correct display."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        self.expect("breakpoint set -f main.c -l 42", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = 42, locations = 1")

        self.runCmd("run", RUN_STOPPED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is Stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # This should display correctly.
        self.expect("variable list bits", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(uint32_t:1) b1 = 0x00000001,',
                       '(uint32_t:2) b2 = 0x00000003,',
                       '(uint32_t:3) b3 = 0x00000007,',
                       '(uint32_t:4) b4 = 0x0000000f,',
                       '(uint32_t:5) b5 = 0x0000001f,',
                       '(uint32_t:6) b6 = 0x0000003f,',
                       '(uint32_t:7) b7 = 0x0000007f,',
                       '(uint32_t:4) four = 0x0000000f'])

        # And so should this.
        # rdar://problem/8348251
        self.expect("variable list", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(uint32_t:1) b1 = 0x00000001,',
                       '(uint32_t:2) b2 = 0x00000003,',
                       '(uint32_t:3) b3 = 0x00000007,',
                       '(uint32_t:4) b4 = 0x0000000f,',
                       '(uint32_t:5) b5 = 0x0000001f,',
                       '(uint32_t:6) b6 = 0x0000003f,',
                       '(uint32_t:7) b7 = 0x0000007f,',
                       '(uint32_t:4) four = 0x0000000f'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
