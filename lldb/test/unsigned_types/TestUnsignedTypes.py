"""
Test that variables with unsigned types display correctly.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *

class TestUnsignedTypes(TestBase):

    mydir = "unsigned_types"

    def test_unsigned_types(self):
        """Test that variables with unsigned types display correctly."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break on line 19 in main() aftre the variables are assigned values.
        self.expect("breakpoint set -f main.cpp -l 19", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = 19, locations = 1")

        self.runCmd("run", RUN_STOPPED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is Stopped', 'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # Test that unsigned types display correctly.
        self.expect("variable list -a", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "the_unsigned_char = (unsigned char) 'c'",
            substrs = ["the_unsigned_short = (short unsigned int) 0x0063",
                       "the_unsigned_int = (unsigned int) 0x00000063",
                       "the_unsigned_long = (long unsigned int) 0x0000000000000063",
                       "the_unsigned_long_long = (long long unsigned int) 0x0000000000000063",
                       "the_uint32 = (uint32_t) 0x00000063"])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
