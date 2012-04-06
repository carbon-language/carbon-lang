"""Test that anonymous structs/unions are transparent to member access"""

import os, time
import unittest2
import lldb
from lldbtest import *

class AnonymousTestCase(TestBase):

    mydir = os.path.join("lang", "c", "anonymous")

    @dsym_test
    def test_expr_with_dsym(self):
        self.buildDsym()
        self.expr()

    @dwarf_test
    def test_expr_with_dwarf(self):
        self.buildDwarf()
        self.expr()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set breakpoint 0 here.')

    def common_setup(self):
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        self.expect("breakpoint set -f main.c -l %d" % self.line, BREAKPOINT_CREATED,
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
        self.expect("expression c->foo.d", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 4"])
            
        self.expect("expression c->b", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 2"])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
