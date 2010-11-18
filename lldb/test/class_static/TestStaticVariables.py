"""
Test display and Python APIs on file and class static variables.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class StaticVariableTestCase(TestBase):

    mydir = "class_static"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_and_run_command(self):
        """Test that file and class static variables display correctly."""
        self.buildDsym()
        self.static_variable_commands()

    def test_with_dwarf_and_run_command(self):
        """Test that anonymous and named namespace variables display correctly."""
        self.buildDwarf()
        self.static_variable_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def static_variable_commands(self):
        """Test that anonymous and named namespace variables display correctly."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f main.cpp -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = %d, locations = 1" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is stopped',
                       'stop reason = breakpoint'])

        # On Mac OS X, gcc 4.2 emits the wrong debug info for A::g_points.
        slist = ['(PointType [2]) g_points', 'A::g_points']

        # 'frame variable -G' finds and displays global variable(s) by name.
        self.expect('frame variable -G g_points', VARIABLES_DISPLAYED_CORRECTLY,
            substrs = slist)

        # A::g_points is an array of two elements.
        if sys.platform.startswith("darwin") and self.getCompiler() in ['clang', 'llvm-gcc']:
            self.expect("frame variable A::g_points[1].x", VARIABLES_DISPLAYED_CORRECTLY,
                startstr = "(int) A::g_points[1].x = 11")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
