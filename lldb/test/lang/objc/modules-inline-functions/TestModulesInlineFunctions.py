"""Test that inline functions from modules are imported correctly"""

import os, time
import unittest2
import lldb
import platform
import lldbutil

from distutils.version import StrictVersion

from lldbtest import *

class ModulesInlineFunctionsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @dsym_test
    def test_expr_with_dsym(self):
        self.buildDsym()
        self.expr()

    @dwarf_test
    @skipIfFreeBSD
    @skipIfLinux
    def test_expr_with_dwarf(self):
        self.buildDwarf()
        self.expr()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.m', '// Set breakpoint here.')

    def applies(self):
        if platform.system() != "Darwin":
            return False
        if StrictVersion('12.0.0') > platform.release():
            return False

        return True

    def common_setup(self):
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_FAILED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

    def expr(self):
        if not self.applies():
            return

        self.common_setup()

        self.runCmd("settings set target.clang-module-search-paths \"" + os.getcwd() + "\"")

        self.expect("expr @import myModule; 3", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["int", "3"])

        self.expect("expr isInline(2)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["4"])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
