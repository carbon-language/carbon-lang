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
        res = self.res
        exe = os.path.join(os.getcwd(), "a.out")
        self.ci.HandleCommand("file " + exe, res)
        self.assertTrue(res.Succeeded(), CURRENT_EXECUTABLE_SET)

        # Break on line 19 in main() aftre the variables are assigned values.
        self.ci.HandleCommand("breakpoint set -f main.cpp -l 19", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().startswith(
            "Breakpoint created: 1: file ='main.cpp', line = 19, locations = 1"
            ),
                        BREAKPOINT_CREATED)

        self.ci.HandleCommand("run", res)
        self.runStarted = True
        self.assertTrue(res.Succeeded(), RUN_STOPPED)

        # The stop reason of the thread should be breakpoint.
        self.ci.HandleCommand("thread list", res)
        self.assertTrue(res.Succeeded(), CMD_MSG('thread list'))
        self.assertTrue(res.GetOutput().find('state is Stopped') > 0 and
                        res.GetOutput().find('stop reason = breakpoint') > 0,
                        STOPPED_DUE_TO_BREAKPOINT)

        # The breakpoint should have a hit count of 1.
        self.ci.HandleCommand("breakpoint list", res)
        self.assertTrue(res.Succeeded())
        self.assertTrue(res.GetOutput().find(' resolved, hit count = 1') > 0,
                        BREAKPOINT_HIT_ONCE)

        # Test that unsigned types display correctly.
        self.ci.HandleCommand("variable list -a", res)
        #print "variable list -a ->", res.GetOutput()
        self.assertTrue(res.Succeeded())
        output = res.GetOutput()
        self.assertTrue(
            output.startswith("the_unsigned_char = (unsigned char) 'c'")
            and
            output.find("the_unsigned_short = (short unsigned int) 0x0063") > 0
            and
            output.find("the_unsigned_int = (unsigned int) 0x00000063") > 0
            and
            output.find("the_unsigned_long = (long unsigned int) "
                        "0x0000000000000063") > 0
            and
            output.find("the_unsigned_long_long = (long long unsigned int)"
                        " 0x0000000000000063") > 0
            and
            output.find("the_uint32 = (uint32_t) 0x00000063"),

            VARIABLES_DISPLAYED_CORRECTLY
            )


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
