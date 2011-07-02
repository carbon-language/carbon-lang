"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class DataFormatterTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "data-formatter-objc")

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
        self.line = line_number('main.m', '// Set break point at this line.')

    def data_formatter_commands(self):
        """Test that that file and class static variables display correctly."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f main.m -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.m', line = %d, locations = 1" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("type summary add -f \"${var%@}\" MyClass")

        self.expect("frame variable object2",
            substrs = ['MyOtherClass']);
        
        self.expect("frame variable *object2",
            substrs = ['MyOtherClass']);

        # Now let's delete the 'MyClass' custom summary.
        self.runCmd("type summary delete MyClass")

        # The type format list should not show 'MyClass' at this point.
        self.expect("type summary list", matching=False,
            substrs = ['MyClass'])

        self.runCmd("type summary add -f \"a test\" MyClass")
        
        self.expect("frame variable object2",
                    substrs = ['a test']);
        
        self.expect("frame variable *object2",
                    substrs = ['a test']);

        self.expect("frame variable object",
                    substrs = ['a test']);
        
        self.expect("frame variable *object",
                    substrs = ['a test']);

        self.runCmd("type summary add -f \"a test\" MyClass -C no")
        
        self.expect("frame variable *object2",
                    substrs = ['*object2 = {',
                               'MyClass = a test',
                               'backup = ']);
        
        self.expect("frame variable object2", matching=False,
                    substrs = ['a test']);
        
        self.expect("frame variable object",
                    substrs = ['a test']);
        
        self.expect("frame variable *object",
                    substrs = ['a test']);


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
