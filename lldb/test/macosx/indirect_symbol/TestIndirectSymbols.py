"""Test stepping and setting breakpoints in indirect and re-exported symbols."""

import os, time
import unittest2
import lldb
import lldbutil
from lldbtest import *

class TestIndirectFunctions(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_with_dsym_and_python_api(self):
        """Test stepping and setting breakpoints in indirect and re-exported symbols."""
        self.buildDsym()
        self.indirect_stepping()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dwarf_test
    def test_with_dwarf_and_python_api(self):
        """Test stepping and setting breakpoints in indirect and re-exported symbols."""
        self.buildDwarf()
        self.indirect_stepping()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = "main.c"

    def indirect_stepping(self):
        """Test stepping and setting breakpoints in indirect and re-exported symbols."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.main_source_spec = lldb.SBFileSpec (self.main_source)

        break1 = target.BreakpointCreateBySourceRegex ("Set breakpoint here to step in indirect.", self.main_source_spec)
        self.assertTrue(break1, VALID_BREAKPOINT)

        break2 = target.BreakpointCreateBySourceRegex ("Set breakpoint here to step in reexported.", self.main_source_spec)
        self.assertTrue(break2, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint (process, break1)
        if len(threads) != 1:
            self.fail ("Failed to stop at breakpoint 1.")

        thread = threads[0]

        # Now do a step-into, and we should end up in the hidden target of this indirect function.
        thread.StepInto()
        curr_function = thread.GetFrameAtIndex(0).GetFunctionName()
        self.assertTrue (curr_function == "call_through_indirect_hidden", "Stepped into indirect symbols.")

        # Now set a breakpoint using the indirect symbol name, and make sure we get to that:
        break_indirect = target.BreakpointCreateByName ("call_through_indirect");
        self.assertTrue (break_indirect, VALID_BREAKPOINT)

        # Now continue should take us to the second call through the indirect symbol:

        threads = lldbutil.continue_to_breakpoint (process, break_indirect)
        self.assertTrue (len(threads) == 1, "Stopped at breakpoint in indirect function.")
        curr_function = thread.GetFrameAtIndex(0).GetFunctionName()
        self.assertTrue (curr_function == "call_through_indirect_hidden", "Stepped into indirect symbols.")

        # Delete this breakpoint so it won't get in the way:
        target.BreakpointDelete (break_indirect.GetID())

        # Now continue to the site of the first re-exported function call in main:
        threads = lldbutil.continue_to_breakpoint (process, break2)

        # This is stepping Into through a re-exported symbol to an indirect symbol:
        thread.StepInto()
        curr_function = thread.GetFrameAtIndex(0).GetFunctionName()
        self.assertTrue (curr_function == "call_through_indirect_hidden", "Stepped into indirect symbols.")

        # And the last bit is to set a breakpoint on the re-exported symbol and make sure we are again in out target function.
        break_reexported = target.BreakpointCreateByName ("reexport_to_indirect");
        self.assertTrue (break_reexported, VALID_BREAKPOINT)

        # Now continue should take us to the second call through the indirect symbol:

        threads = lldbutil.continue_to_breakpoint (process, break_reexported)
        self.assertTrue (len(threads) == 1, "Stopped at breakpoint in reexported function target.")
        curr_function = thread.GetFrameAtIndex(0).GetFunctionName()
        self.assertTrue (curr_function == "call_through_indirect_hidden", "Stepped into indirect symbols.")


 
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
