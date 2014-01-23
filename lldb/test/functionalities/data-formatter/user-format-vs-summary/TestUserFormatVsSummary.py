"""
Test that the user can input a format and it will prevail over summary format's choices.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class UserFormatVSSummaryTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test that the user can input a format and it will prevail over summary format's choices."""
        self.buildDsym()
        self.data_formatter_commands()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Test that the user can input a format and it will prevail over summary format's choices."""
        self.buildDwarf()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def data_formatter_commands(self):
        """Test that the user can input a format and it will prevail over summary format's choices."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        self.expect("frame variable p1", substrs = ['(Pair) p1 = (x = 3, y = -3)']);

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd('type summary add Pair -s "x=${var.x%d},y=${var.y%u}"')

        self.expect("frame variable p1", substrs = ['(Pair) p1 = x=3,y=4294967293']);
        self.expect("frame variable -f x p1", substrs = ['(Pair) p1 = x=0x00000003,y=0xfffffffd']);
        self.expect("frame variable -f d p1", substrs = ['(Pair) p1 = x=3,y=-3']);
        self.expect("frame variable p1", substrs = ['(Pair) p1 = x=3,y=4294967293']);

        self.runCmd('type summary add Pair -s "x=${var.x%x},y=${var.y%u}"')

        self.expect("frame variable p1", substrs = ['(Pair) p1 = x=0x00000003,y=4294967293']);
        self.expect("frame variable -f d p1", substrs = ['(Pair) p1 = x=3,y=-3']);

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
