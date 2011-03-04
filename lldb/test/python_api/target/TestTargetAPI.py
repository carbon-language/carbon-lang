"""
Test SBTarget APIs.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class TargetAPITestCase(TestBase):

    mydir = os.path.join("python_api", "target")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_resolve_symbol_context_with_address_with_dsym(self):
        """Exercise SBTaget.ResolveSymbolContextForAddress() API."""
        self.buildDsym()
        self.resolve_symbol_context_with_address()

    @python_api_test
    def test_resolve_symbol_context_with_address_with_dwarf(self):
        """Exercise SBTarget.ResolveSymbolContextForAddress() API."""
        self.buildDwarf()
        self.resolve_symbol_context_with_address()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line1 = line_number('main.c', '// Find the line number for breakpoint 1 here.')
        self.line2 = line_number('main.c', '// Find the line number for breakpoint 2 here.')

    def resolve_symbol_context_with_address(self):
        """Exercise SBTaget.ResolveSymbolContextForAddress() API."""
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
        error = lldb.SBError()
        self.process = target.Launch (self.dbg.GetListener(), None, None, os.ctermid(), os.ctermid(), os.ctermid(), None, 0, False, error)

        self.process = target.GetProcess()
        self.assertTrue(self.process.IsValid(), PROCESS_IS_VALID)

        # Frame #0 should be on self.line1.
        self.assertTrue(self.process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint condition")
        #self.runCmd("process status")
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertTrue(lineEntry.GetLine() == self.line1)

        address1 = lineEntry.GetStartAddress()

        # Continue the inferior, the breakpoint 2 should be hit.
        self.process.Continue()
        self.assertTrue(self.process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint condition")
        #self.runCmd("process status")
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertTrue(lineEntry.GetLine() == self.line2)

        address2 = lineEntry.GetStartAddress()

        #print "address1:", address1
        #print "address2:", address2

        # Now call SBTarget.ResolveSymbolContextForAddress() with the addresses from our line entry.
        context1 = target.ResolveSymbolContextForAddress(address1, lldb.eSymbolContextEverything)
        context2 = target.ResolveSymbolContextForAddress(address2, lldb.eSymbolContextEverything)

        self.assertTrue(context1.IsValid() and context2.IsValid())
        #print "context1:", context1
        #print "context2:", context2

        # Verify that the context point to the same function 'a'.
        symbol1 = context1.GetSymbol()
        symbol2 = context2.GetSymbol()
        self.assertTrue(symbol1.IsValid() and symbol2.IsValid())
        #print "symbol1:", symbol1
        #print "symbol2:", symbol2

        stream1 = lldb.SBStream()
        symbol1.GetDescription(stream1)
        stream2 = lldb.SBStream()
        symbol2.GetDescription(stream2)
        
        self.expect(stream1.GetData(), "The two addresses should resolve to the same symbol", exe=False,
            startstr = stream2.GetData())

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
