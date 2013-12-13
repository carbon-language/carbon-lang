"""Test that the 'add-dsym', aka 'target symbols add', succeeds in the middle of debug session."""

import os, time
import unittest2
import lldb
import pexpect
from lldbtest import *

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class AddDsymMidExecutionCommandCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.source = 'main.c'

    def test_add_dsym_mid_execution(self):
        """Test that add-dsym mid-execution loads the symbols at the right place for a slid binary."""
        self.buildDsym(clean=True)
        exe = os.path.join(os.getcwd(), "a.out")

        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        main_bp = self.target.BreakpointCreateByName ("main", "a.out")
        self.assertTrue(main_bp, VALID_BREAKPOINT)

        self.runCmd("settings set target.disable-aslr false")
        self.process = self.target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(self.process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.assertTrue(self.process.GetState() == lldb.eStateStopped,
                        STOPPED_DUE_TO_BREAKPOINT)

        self.runCmd("add-dsym hide.app/Contents/a.out.dSYM")

        self.expect("frame select",
                    substrs = ['a.out`main at main.c'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
