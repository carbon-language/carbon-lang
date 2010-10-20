"""
Test breakpoint conditions with 'breakpoint modify -c <expr> id'.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class BreakpointConditionsTestCase(TestBase):

    mydir = "breakpoint_conditions"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_python(self):
        """Exercise breakpoint condition with 'breakpoint modify -c <expr> id'."""
        self.buildDsym()
        self.breakpoint_conditions()

    def test_with_dwarf_python(self):
        """Exercise breakpoint condition with 'breakpoint modify -c <expr> id'."""
        self.buildDwarf()
        self.breakpoint_conditions()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line1 = line_number('main.c', '// Find the line number of function "c" here.')
        self.line2 = line_number('main.c', "// Find the line number of c's parent call here.")

    def breakpoint_conditions(self):
        """Exercise breakpoint condition with 'breakpoint modify -c <expr> id'."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Create a breakpoint by function name 'c'.
        self.expect("breakpoint set -n c", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: name = 'c', locations = 1")

        # And set a condition on the breakpoint to stop on when 'val == 3'.
        self.runCmd("breakpoint modify -c 'val == 3' 1")

        # Now run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # 'frame variable -t val' should return 3 due to breakpoint condition.
        self.expect("frame variable -t val", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = '(int) val = 3')

        # Also check the hit count, which should be 3, by design.
        self.expect("breakpoint list", BREAKPOINT_HIT_THRICE,
            substrs = ["resolved = 1",
                       "Condition: val == 3",
                       "hit count = 3"])

        # The frame #0 should correspond to main.c:36, the executable statement
        # in function name 'c'.  And the parent frame should point to main.c:24.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT_CONDITION,
            #substrs = ["stop reason = breakpoint"],
            patterns = ["frame #0.*main.c:%d" % self.line1,
                        "frame #1.*main.c:%d" % self.line2])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
