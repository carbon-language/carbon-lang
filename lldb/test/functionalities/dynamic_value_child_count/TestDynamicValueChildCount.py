"""
Test that dynamic values update their child count correctly
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class DynamicValueChildCountTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    @expectedFailurei386("to be figured out")
    def test_get_dynamic_vals_with_dsym(self):
        """Test fetching C++ dynamic values from pointers & references."""
        self.buildDsym(dictionary=self.getBuildFlags())
        self.do_get_dynamic_vals()

    @expectedFailureLinux # FIXME: This needs to be root-caused.  It looks like the DWARF info is anticipating the derived class assignment.
    @expectedFailureFreeBSD("llvm.org/pr19311") # continue at a breakpoint does not work
    @python_api_test
    @dwarf_test
    @expectedFailurei386("to be figured out")
    def test_get_dynamic_vals_with_dwarf(self):
        """Test fetching C++ dynamic values from pointers & references."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.do_get_dynamic_vals()

    def setUp(self):
        # Call super's setUp().                                                                                                           
        TestBase.setUp(self)

        # Find the line number to break for main.c.                                                                                       

        self.main_third_call_line = line_number('pass-to-base.cpp',
                                                       '// Break here and check b has 0 children')
        self.main_fourth_call_line = line_number('pass-to-base.cpp',
                                                       '// Break here and check b still has 0 children')
        self.main_fifth_call_line = line_number('pass-to-base.cpp',
                                                       '// Break here and check b has one child now')
        self.main_sixth_call_line = line_number('pass-to-base.cpp',
                                                       '// Break here and check b has 0 children again')




    def do_get_dynamic_vals(self):
        """Get argument vals for the call stack when stopped on a breakpoint."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target from the debugger.

        target = self.dbg.CreateTarget (exe)
        self.assertTrue(target, VALID_TARGET)

        # Set up our breakpoints:

        third_call_bpt = target.BreakpointCreateByLocation('pass-to-base.cpp', self.main_third_call_line)
        self.assertTrue(third_call_bpt,
                        VALID_BREAKPOINT)
        fourth_call_bpt = target.BreakpointCreateByLocation('pass-to-base.cpp', self.main_fourth_call_line)
        self.assertTrue(fourth_call_bpt,
                        VALID_BREAKPOINT)
        fifth_call_bpt = target.BreakpointCreateByLocation('pass-to-base.cpp', self.main_fifth_call_line)
        self.assertTrue(fifth_call_bpt,
                        VALID_BREAKPOINT)
        sixth_call_bpt = target.BreakpointCreateByLocation('pass-to-base.cpp', self.main_sixth_call_line)
        self.assertTrue(sixth_call_bpt,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        b = self.frame().FindVariable("b").GetDynamicValue(lldb.eDynamicCanRunTarget)
        self.assertTrue(b.GetNumChildren() == 0, "b has 0 children")
        self.runCmd("continue")
        self.assertTrue(b.GetNumChildren() == 0, "b still has 0 children")
        self.runCmd("continue")
        self.assertTrue(b.GetNumChildren() != 0, "b now has 1 child")
        self.runCmd("continue")
        self.assertTrue(b.GetNumChildren() == 0, "b didn't go back to 0 children")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
