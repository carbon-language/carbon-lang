"""
Test breakpoint command for different options.
"""

import os
import unittest2
import lldb
from lldbtest import *
import lldbutil

class BreakpointOptionsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @dsym_test
    def test_with_dsym(self):
        """Test breakpoint command for different options."""
        self.buildDsym()
        self.breakpoint_options_test()

    @dwarf_test
    def test_with_dwarf(self):
        """Test breakpoint command for different options."""
        self.buildDwarf()
        self.breakpoint_options_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    def breakpoint_options_test(self):
        """Test breakpoint command for different options."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint with 1 locations.
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, extra_options = "-K 1", num_expected_locations = 1)
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, extra_options = "-K 0", num_expected_locations = 1)

        # This should create a breakpoint 0 locations.
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, extra_options = "-m 0", num_expected_locations = 0)

        # Run the program.
        self.runCmd("run", RUN_FAILED)

        # Stopped once.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ["stop reason = breakpoint 2."])

        # Check the list of breakpoint.
        self.expect("breakpoint list -f", "Breakpoint locations shown correctly",
            substrs = ["1: file = 'main.c', line = %d, exact_match = 0, locations = 1" % self.line,
                       "2: file = 'main.c', line = %d, exact_match = 0, locations = 1" % self.line,
                       "3: file = 'main.c', line = %d, exact_match = 1, locations = 0" % self.line])

        # Continue the program, there should be another stop.
        self.runCmd("process continue")

        # Stopped again.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ["stop reason = breakpoint 1."])

        # Continue the program, we should exit.
        self.runCmd("process continue")

        # We should exit.
        self.expect("process status", "Process exited successfully",
            patterns = ["^Process [0-9]+ exited with status = 0"])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
