"""
Test more expression command sequences with objective-c.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class FoundationTestCase2(TestBase):

    mydir = "foundation"

    def test_more_expr_commands_with_dsym(self):
        """More expression commands for objective-c."""
        self.buildDsym()
        self.more_expr_objc()

    def test_more_expr_commands_with_dwarf(self):
        """More expression commands for objective-c."""
        self.buildDwarf()
        self.more_expr_objc()

    def test_NSArray_expr_commands_with_dsym(self):
        """Test expression commands for NSArray."""
        self.buildDsym()
        self.NSArray_expr()

    def test_NSArray_expr_commands_with_dwarf(self):
        """Test expression commands for NSArray."""
        self.buildDwarf()
        self.NSArray_expr()

    def test_NSString_expr_commands_with_dsym(self):
        """Test expression commands for NSString."""
        self.buildDsym()
        self.NSString_expr()

    def test_NSString_expr_commands_with_dwarf(self):
        """Test expression commands for NSString."""
        self.buildDwarf()
        self.NSString_expr()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break at.
        self.lines = []
        self.lines.append(line_number('main.m', '// Expressions to test here for selector:'))
        self.lines.append(line_number('main.m', '// Expressions to test here for NSArray:'))
        self.lines.append(line_number('main.m', '// Expressions to test here for NSString:'))
        self.lines.append(line_number('main.m', "// Set a breakpoint on '-[MyString description]' and test expressions:"))

    def more_expr_objc(self):
        """More expression commands for objective-c."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Create a bunch of breakpoints.
        for line in self.lines:
            self.expect("breakpoint set -f main.m -l %d" % line, BREAKPOINT_CREATED,
                substrs = ["Breakpoint created:",
                           "file ='main.m', line = %d, locations = 1" % line])

        self.runCmd("run", RUN_SUCCEEDED)

        # Test_Selector:
        self.runCmd("thread backtrace")
        self.expect("expression (char *)sel_getName(sel)",
            substrs = ["(char *)",
                       "length"])

        self.runCmd("process continue")

        # Test_NSArray:
        self.runCmd("thread backtrace")
        self.runCmd("process continue")

        # Test_NSString:
        self.runCmd("thread backtrace")
        self.runCmd("process continue")

        # Test_MyString:
        self.runCmd("thread backtrace")
        self.expect("expression (char *)sel_getName(_cmd)",
            substrs = ["(char *)",
                       "description"])

        self.runCmd("process continue")

    @unittest2.expectedFailure
    def NSArray_expr(self):
        """Test expression commands for NSArray."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside Test_NSArray:
        line = self.lines[1]
        self.expect("breakpoint set -f main.m -l %d" % line, BREAKPOINT_CREATED,
            substrs = ["Breakpoint created:",
                       "file ='main.m', line = %d, locations = 1" % line])

        self.runCmd("run", RUN_SUCCEEDED)

        # Test_NSArray:
        self.runCmd("thread backtrace")
        self.expect("expression [array1 count]")
        self.expect("expression array1.count")
        self.expect("expression [array2 count]")
        self.expect("expression array2.count")
        self.runCmd("process continue")

    @unittest2.expectedFailure
    def NSString_expr(self):
        """Test expression commands for NSString."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside Test_NSString:
        line = self.lines[2]
        self.expect("breakpoint set -f main.m -l %d" % line, BREAKPOINT_CREATED,
            substrs = ["Breakpoint created:",
                       "file ='main.m', line = %d, locations = 1" % line])

        self.runCmd("run", RUN_SUCCEEDED)

        # Test_NSString:
        self.runCmd("thread backtrace")
        self.expect("expression [str length]")
        self.expect("expression [id length]")
        self.expect("expression [str description]")
        self.expect("expression [id description]")
        self.expect("expression str.description")
        self.expect("expression id.description")
        self.expect('expression str = @"new"')
        self.expect('expression str = [NSString stringWithFormat: @"%cew", \'N\']')
        self.runCmd("process continue")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
