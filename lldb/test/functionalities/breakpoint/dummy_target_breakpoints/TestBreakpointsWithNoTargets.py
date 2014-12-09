"""
Test breakpoint commands set before we have a target
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class BreakpointInDummyTarget (TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test breakpoint set before we have a target. """
        self.buildDsym()
        self.dummy_breakpoint_test()

    @dwarf_test
    def test_with_dwarf(self):
        """Test breakpoint set before we have a target. """
        self.buildDwarf()
        self.dummy_breakpoint_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', 'Set a breakpoint on this line.')
        self.line2 = line_number ('main.c', 'Set another on this line.')

    def dummy_breakpoint_test(self):
        """Test breakpoint set before we have a target. """

        # This should create a breakpoint with 3 locations.
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=0)
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line2, num_expected_locations=0)
        
        # This is the function to remove breakpoints from the dummy target
        # to get a clean slate for the next test case.
        def cleanup():
            self.runCmd('breakpoint delete -D -f', check=False)
            self.runCmd('breakpoint list', check=False)


        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)
        
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # The breakpoint list should show 3 locations.
        self.expect("breakpoint list -f", "Breakpoint locations shown correctly",
            substrs = ["1: file = 'main.c', line = %d, locations = 1" % self.line,
                       "2: file = 'main.c', line = %d, locations = 1" % self.line2])

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Stopped once.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ["stop reason = breakpoint 1."])

        # Continue the program, there should be another stop.
        self.runCmd("process continue")

        # Stopped again.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ["stop reason = breakpoint 2."])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
