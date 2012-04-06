"""Test that hidden ivars in a shared library are visible from the main executable."""

import os, time
import unittest2
import lldb
from lldbtest import *

class HiddenIvarsTestCase(TestBase):

    mydir = os.path.join("lang", "objc", "hidden-ivars")

    @dsym_test
    def test_expr_with_dsym(self):
        self.buildDsym()
        self.expr()

    @dwarf_test
    def test_expr_with_dwarf(self):
        self.buildDwarf()
        self.expr()

    @dsym_test
    def test_frame_variable_with_dsym(self):
        self.buildDsym()
        self.frame_var()

    @dwarf_test
    def test_frame_variable_with_dwarf(self):
        self.buildDwarf()
        self.frame_var()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.m', '// Set breakpoint 0 here.')

    def common_setup(self):
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        self.expect("breakpoint set -f main.m -l %d" % self.line, BREAKPOINT_CREATED,
            startstr = "Breakpoint created")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

    def expr(self):
        self.common_setup()

        # This should display correctly.
        self.expect("expression (j->_definer->bar)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 5"])
            
        self.expect("expression *(j->_definer)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["foo = 0", "bar = 5"])

    def frame_var(self):
        self.common_setup()

        # This should display correctly.
        self.expect("frame variable j->_definer->bar", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 5"])
            
        self.expect("frame variable *j->_definer", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["foo = 0", "bar = 5"])
                       
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
