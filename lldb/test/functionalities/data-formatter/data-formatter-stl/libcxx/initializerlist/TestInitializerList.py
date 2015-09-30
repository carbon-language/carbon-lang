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

    @skipIfWindows # libc++ not ported to Windows yet
    @skipIfGcc
    @expectedFailureLinux # fails on clang 3.5 and tot
    def test(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.skip_if_library_missing(self, self.target(), lldbutil.PrintableRegex("libc\+\+"))

        bkpt = self.target().FindBreakpointByID(lldbutil.run_break_set_by_source_regexp (self, "Set break point at this line."))

        self.runCmd("run", RUN_SUCCEEDED)

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
