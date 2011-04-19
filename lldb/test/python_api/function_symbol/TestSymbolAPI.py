"""
Test newly added SBSymbol and SBAddress APIs.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class SymbolAPITestCase(TestBase):

    mydir = os.path.join("python_api", "function_symbol")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_with_dsym(self):
        """Exercise some SBSymbol and SBAddress APIs."""
        self.buildDsym()
        self.symbol_and_address_api()

    @python_api_test
    def test_with_dwarf(self):
        """Exercise some SBSymbol and SBAddress APIs."""
        self.buildDwarf()
        self.symbol_and_address_api()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line1 = line_number('main.c', '// Find the line number for breakpoint 1 here.')
        self.line2 = line_number('main.c', '// Find the line number for breakpoint 2 here.')

    def symbol_and_address_api(self):
        """Exercise some SBSymbol and SBAddress APIs."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Now create the two breakpoints inside function 'a'.
        breakpoint1 = target.BreakpointCreateByLocation('main.c', self.line1)
        breakpoint2 = target.BreakpointCreateByLocation('main.c', self.line2)
        #print "breakpoint1:", breakpoint1
        #print "breakpoint2:", breakpoint2
        self.assertTrue(breakpoint1.IsValid() and
                        breakpoint1.GetNumLocations() == 1,
                        VALID_BREAKPOINT)
        self.assertTrue(breakpoint2.IsValid() and
                        breakpoint2.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        self.process = target.LaunchSimple(None, None, os.getcwd())

        self.process = target.GetProcess()
        self.assertTrue(self.process.IsValid(), PROCESS_IS_VALID)

        # Frame #0 should be on self.line1.
        self.assertTrue(self.process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)
        symbol_line1 = frame0.GetSymbol()
        # We should have a symbol type of code.
        self.assertTrue(symbol_line1.GetType() == lldb.eSymbolTypeCode)
        addr_line1 = symbol_line1.GetStartAddress()
        # And a section type of code, too.
        self.assertTrue(addr_line1.GetSectionType() == lldb.eSectionTypeCode)

        # Continue the inferior, the breakpoint 2 should be hit.
        self.process.Continue()
        self.assertTrue(self.process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)
        symbol_line2 = frame0.GetSymbol()
        # We should have a symbol type of code.
        self.assertTrue(symbol_line2.GetType() == lldb.eSymbolTypeCode)
        addr_line2 = symbol_line2.GetStartAddress()
        # And a section type of code, too.
        self.assertTrue(addr_line2.GetSectionType() == lldb.eSectionTypeCode)

        # Now verify that both addresses point to the same module.
        if self.TraceOn():
            print "UUID:", addr_line1.GetModule().GetUUIDString()
        self.assertTrue(addr_line1.GetModule().GetUUIDString() == addr_line2.GetModule().GetUUIDString())

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
