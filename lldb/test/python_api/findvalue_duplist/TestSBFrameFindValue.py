"""Test that SBFrame::FindValue finds things but does not duplicate the entire variables list"""

import os, sys, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class SBFrameFindValueTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_with_dsym_formatters_api(self):
        """Test that SBFrame::FindValue finds things but does not duplicate the entire variables list"""
        self.buildDsym()
        self.setTearDownCleanup()
        self.commands()

    @python_api_test
    @dwarf_test
    def test_with_dwarf_formatters_api(self):
        """Test that SBFrame::FindValue finds things but does not duplicate the entire variables list"""
        self.buildDwarf()
        self.setTearDownCleanup()
        self.commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def commands(self):
        """Test that SBFrame::FindValue finds things but does not duplicate the entire variables list"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        
        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex('Set breakpoint here', lldb.SBFileSpec("main.cpp"))
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint (process, breakpoint)
        
        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        self.assertTrue(self.frame.GetVariables(True,True,False,True).GetSize() == 2, "variable count is off")
        self.assertFalse(self.frame.FindValue("NoSuchThing",lldb.eValueTypeVariableArgument,lldb.eDynamicCanRunTarget).IsValid(), "found something that should not be here")
        self.assertTrue(self.frame.GetVariables(True,True,False,True).GetSize() == 2, "variable count is off after failed FindValue()")
        self.assertTrue(self.frame.FindValue("a",lldb.eValueTypeVariableArgument,lldb.eDynamicCanRunTarget).IsValid(), "FindValue() didn't find an argument")
        self.assertTrue(self.frame.GetVariables(True,True,False,True).GetSize() == 2, "variable count is off after successful FindValue()")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
