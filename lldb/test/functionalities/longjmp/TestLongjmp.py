"""
Test the use of setjmp/longjmp for non-local goto operations in a single-threaded inferior.
"""

import os
import unittest2
import lldb
import lldbutil
from lldbtest import *

class LongjmpTestCase(TestBase):

    mydir = os.path.join("functionalities", "longjmp")

    def setUp(self):
        TestBase.setUp(self)

    def test_step_out(self):
        """Test stepping when the inferior calls setjmp/longjmp, in particular, thread step-out."""
        self.buildDefault()
        self.step_out()

    def test_step_over(self):
        """Test stepping when the inferior calls setjmp/longjmp, in particular, thread step-over a longjmp."""
        self.buildDefault()
        self.step_over()

    def test_step_back_out(self):
        """Test stepping when the inferior calls setjmp/longjmp, in particular, thread step-out after thread step-in."""
        self.buildDefault()
        self.step_back_out()

    def start_test(self, symbol):
        exe = os.path.join(os.getcwd(), "a.out")

        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break in main().
        lldbutil.run_break_set_by_symbol (self, symbol, num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped', 'stop reason = breakpoint'])

    def step_out(self):
        self.start_test("do_jump")

        self.runCmd("thread step-out", RUN_SUCCEEDED)
        self.expect("process status", PROCESS_STOPPED,
            patterns = ['Process .* exited with status = 0'])

    def step_over(self):
        self.start_test("do_jump")

        self.runCmd("thread step-over", RUN_SUCCEEDED)
        self.expect("process status", PROCESS_STOPPED,
            patterns = ['Process .* exited with status = 0'])

    def step_back_out(self):
        self.start_test("main")
        
        self.runCmd("thread step-over", RUN_SUCCEEDED)
        self.runCmd("thread step-in", RUN_SUCCEEDED)
        self.runCmd("thread step-out", RUN_SUCCEEDED)
        self.expect("process status", PROCESS_STOPPED,
            patterns = ['Process .* exited with status = 0'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
