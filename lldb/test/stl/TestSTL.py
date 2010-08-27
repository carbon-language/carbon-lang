"""
Test that we can successfully step into an STL function.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestSTL(TestBase):

    mydir = "stl"

    @unittest2.expectedFailure
    def test_step_into_stl(self):
        """Test that we can successfully step into an STL function."""
        exe = os.path.join(os.getcwd(), "a.out")

        # The following two lines, if uncommented, will enable loggings.
        #self.ci.HandleCommand("log enable -f /tmp/lldb.log lldb default", res)
        #self.assertTrue(res.Succeeded())

        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break on line 13 of main.cpp.
        self.expect("breakpoint set -f main.cpp -l 13", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = 13, locations = 1")

        self.runCmd("run", RUN_SUCCEEDED)

        # Stop at 'std::string hello_world ("Hello World!");'.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['main.cpp:13',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # Now do 'thread step-in', we should stop on the basic_string template.
        #
        # This assertion currently always fails.
        # This might be related: rdar://problem/8247112.
        #
        #self.runCmd("thread step-in", trace=True)
        self.runCmd("thread step-in")

        self.expect("thread backtrace", "We have stepped in STL",
             substrs = ['[inlined]',
                        'basic_string.h'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
