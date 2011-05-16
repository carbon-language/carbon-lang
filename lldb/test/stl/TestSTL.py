"""
Test that we can successfully step into an STL function.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class STLTestCase(TestBase):

    mydir = "stl"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym(self):
        """Test that we can successfully step into an STL function."""
        self.buildDsym()
        self.step_into_stl()

    def test_with_dwarf(self):
        """Test that we can successfully step into an STL function."""
        self.buildDwarf()
        self.step_into_stl()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def step_into_stl(self):
        """Test that we can successfully step into an STL function."""
        exe = os.path.join(os.getcwd(), "a.out")

        # The following two lines, if uncommented, will enable loggings.
        #self.ci.HandleCommand("log enable -f /tmp/lldb.log lldb default", res)
        #self.assertTrue(res.Succeeded())

        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # rdar://problem/8543077
        # test/stl: clang built binaries results in the breakpoint locations = 3,
        # is this a problem with clang generated debug info?
        #
        # Break on line 13 of main.cpp.
        self.expect("breakpoint set -f main.cpp -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = %d" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        # Stop at 'std::string hello_world ("Hello World!");'.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['main.cpp:%d' % self.line,
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # Now do 'thread step-in', if we have successfully stopped, we should
        # stop due to the reason of "step in".
        self.runCmd("thread step-in")

        self.runCmd("process status")
        if "stopped" in self.res.GetOutput():
            self.expect("thread backtrace", "We have successfully stepped in",
                        substrs = ['stop reason = step in'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
