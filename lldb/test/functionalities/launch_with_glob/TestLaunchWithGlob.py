"""
Test that argdumper is a viable launching strategy.
"""
import commands
import lldb
import os
import time
import unittest2
from lldbtest import *
import lldbutil

class LaunchWithGlobTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

        
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym (self):
        self.buildDsym()
        self.do_test ()


    @expectedFailureFreeBSD("llvm.org/pr22627 process launch w/ glob not working")
    @dwarf_test
    def test_with_dwarf (self):
        self.buildDwarf()
        self.do_test ()

    def do_test (self):
        exe = os.path.join (os.getcwd(), "a.out")
        
        self.runCmd("target create %s" % exe)
        
        # Create the target
        target = self.dbg.CreateTarget(exe)
        
        # Create any breakpoints we need
        breakpoint = target.BreakpointCreateBySourceRegex ('break here', lldb.SBFileSpec ("main.cpp", False))
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        self.runCmd("process launch -G true -w %s -- fi*.tx?" % (os.getcwd()))

        process = self.process()

        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        STOPPED_DUE_TO_BREAKPOINT)

        thread = process.GetThreadAtIndex (0)

        self.assertTrue (thread.IsValid(),
                         "Process stopped at 'main' should have a valid thread");

        stop_reason = thread.GetStopReason()
        
        self.assertTrue (stop_reason == lldb.eStopReasonBreakpoint,
                         "Thread in process stopped in 'main' should have a stop reason of eStopReasonBreakpoint");

        self.expect("frame variable argv[1]", substrs=['file1.txt'])
        self.expect("frame variable argv[2]", substrs=['file2.txt'])
        self.expect("frame variable argv[3]", substrs=['file3.txt'])
        self.expect("frame variable argv[4]", substrs=['file4.txy'])
        self.expect("frame variable argv[5]", substrs=['file5.tyx'], matching=False)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

