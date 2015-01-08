"""
Verify that the hash computing logic for ValueObject's values can't crash us.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class ValueMD5CrashTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Verify that the hash computing logic for ValueObject's values can't crash us."""
        self.buildDsym()
        self.doThings()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Verify that the hash computing logic for ValueObject's values can't crash us."""
        self.buildDwarf()
        self.doThings()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// break here')

    def doThings(self):
        """Verify that the hash computing logic for ValueObject's values can't crash us."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        value = self.frame().FindVariable("a")
        value.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        
        v = value.GetValue()
        self.assertTrue(value.GetTypeName() == "B *", "a is a B*")
        
        self.runCmd("next")
        self.runCmd("process kill")
        
        # now the process is dead, and value needs updating
        v = value.GetValue()
        
        # if we are here, instead of crashed, the test succeeded

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
