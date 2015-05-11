"""
Test continue from a breakpoint when there is a breakpoint on the next instruction also.
"""

import unittest2
import lldb, lldbutil
from lldbtest import *

class ConsecutiveBreakpoitsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @dsym_test
    @unittest2.expectedFailure("llvm.org/pr23478")
    def test_with_dsym (self):
        self.buildDsym ()
        self.consecutive_breakpoints_tests()

    @dwarf_test
    @unittest2.expectedFailure("llvm.org/pr23478")
    def test_with_dwarf (self):
        self.buildDwarf ()
        self.consecutive_breakpoints_tests()

    def consecutive_breakpoints_tests(self):
        exe = os.path.join (os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateBySourceRegex("Set breakpoint here", lldb.SBFileSpec("main.cpp"))
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # We should be stopped at the first breakpoint
        thread = process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonBreakpoint)

        # Step to the next instruction
        thread.StepInstruction(False)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonPlanComplete)
        address = thread.GetFrameAtIndex(0).GetPC()

        # Run the process until termination
        process.Continue()

        # Now launch the process again, and do not stop at entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # We should be stopped at the first breakpoint
        thread = process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonBreakpoint)

        # Set breakpoint to the next instruction
        target.BreakpointCreateByAddress(address)
        process.Continue()

        # We should be stopped at the second breakpoint
        thread = process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonBreakpoint)

        # Run the process until termination
        process.Continue()

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

