"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class InitializerListTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @dsym_test
    def test_with_dsym(self):
        """Test data formatter commands."""
        self.buildDsym()
        self.data_formatter_commands()

    @skipIfWindows # libc++ not ported to Windows yet
    @skipIfGcc
    @expectedFailureLinux # fails on clang 3.5 and tot
    @dwarf_test
    def test_with_dwarf(self):
        """Test data formatter commands."""
        self.buildDwarf()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def data_formatter_commands(self):
        """Test that that file and class static variables display correctly."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        bkpt = self.target().FindBreakpointByID(lldbutil.run_break_set_by_source_regexp (self, "Set break point at this line."))

        self.runCmd("run", RUN_FAILED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        self.expect("frame variable ili", substrs = ['[1] = 2','[4] = 5'])
        self.expect("frame variable ils", substrs = ['[4] = "surprise it is a long string!! yay!!"'])

        self.expect('image list', substrs = self.getLibcPlusPlusLibs())

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
