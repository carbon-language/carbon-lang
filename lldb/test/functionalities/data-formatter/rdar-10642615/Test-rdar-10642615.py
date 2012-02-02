"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class Radar10642615DataFormatterTestCase(TestBase):

    # test for rdar://problem/10642615 ()
    mydir = os.path.join("functionalities", "data-formatter", "rdar-10642615")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_and_run_command(self):
        """Test data formatter commands."""
        self.buildDsym()
        self.data_formatter_commands()

    def test_with_dwarf_and_run_command(self):
        """Test data formatter commands."""
        self.buildDwarf()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def data_formatter_commands(self):
        """Test that that file and class static variables display correctly."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f main.cpp -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = %d" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect('frame variable value',
            substrs = ['[0] = 1', '[2] = 4'])

        self.runCmd("type summary add vFloat --inline-children")
     
        self.expect('frame variable value',
            substrs = ['[0]=1, [1]', ', [2]=4'])

        self.runCmd("type summary add vFloat --inline-children --omit-names")

        self.expect('frame variable value',
            substrs = ['1, 0, 4, 0'])

        self.runCmd("type summary add vFloat --inline-children")

        self.expect('frame variable value',
	            substrs = ['[0]=1, [1]', ', [2]=4'])

        self.runCmd("type summary delete vFloat")

        self.expect('frame variable value',
            substrs = ['[0] = 1', '[2] = 4'])

        self.runCmd("type summary add vFloat --omit-names", check=False) # should not work since we're not inlining children

        self.expect('frame variable value',
            substrs = ['[0] = 1', '[2] = 4'])

        self.runCmd("type summary add vFloat --inline-children --omit-names")

        self.expect('frame variable value',
            substrs = ['1, 0, 4, 0'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
