"""
Test SBBreakpoint APIs.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class BreakpointAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_breakpoint_is_valid_with_dsym(self):
        """Make sure that if an SBBreakpoint gets deleted its IsValid returns false."""
        self.buildDsym()
        self.breakpoint_is_valid()

    @python_api_test
    @dwarf_test
    def test_breakpoint_is_valid_with_dwarf(self):
        """Make sure that if an SBBreakpoint gets deleted its IsValid returns false."""
        self.buildDwarf()
        self.breakpoint_is_valid ()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def breakpoint_is_valid(self):
        """Get an SBBreakpoint object, delete it from the target and make sure it is no longer valid."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'AFunction'.
        breakpoint = target.BreakpointCreateByName('AFunction', 'a.out')
        #print "breakpoint:", breakpoint
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now delete it:
        did_delete = target.BreakpointDelete(breakpoint.GetID())
        self.assertTrue (did_delete, "Did delete the breakpoint we just created.")

        # Make sure we can't find it:
        del_bkpt = target.FindBreakpointByID (breakpoint.GetID())
        self.assertTrue (not del_bkpt, "We did delete the breakpoint.")

        # Finally make sure the original breakpoint is no longer valid.
        self.assertTrue (not breakpoint, "Breakpoint we deleted is no longer valid.")

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
