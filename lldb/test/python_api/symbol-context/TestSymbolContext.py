"""
Test SBSymbolContext APIs.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class SymbolContextAPITestCase(TestBase):

    mydir = os.path.join("python_api", "symbol-context")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_with_dsym(self):
        """Exercise SBSymbolContext API extensively."""
        self.buildDsym()
        self.symbol_context()

    @python_api_test
    def test_with_dwarf(self):
        """Exercise SBSymbolContext API extensively."""
        self.buildDwarf()
        self.symbol_context()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line = line_number('main.c', '// Find the line number of function "c" here.')

    def symbol_context(self):
        """Get an SBSymbolContext object and call its many methods."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        #print "breakpoint:", breakpoint
        self.assertTrue(breakpoint.IsValid() and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        self.process = target.LaunchProcess([''], [''], os.ctermid(), 0, False)

        self.process = target.GetProcess()
        self.assertTrue(self.process.IsValid(), PROCESS_IS_VALID)

        # Frame #0 should be on self.line.
        frame0 = self.process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        self.assertTrue(frame0.GetLineEntry().GetLine() == self.line)

        # Now get the SBSymbolContext from this frame.  We want everything. :-)
        context = frame0.GetSymbolContext(lldb.eSymbolContextEverything)
        self.assertTrue(context.IsValid())

        module = context.GetModule()
        self.expect(repr(module), "The module should match", exe=False,
            substrs = [os.path.join(self.mydir, 'a.out')])
        #print "module:", module

        compileUnit = context.GetCompileUnit()
        self.expect(repr(compileUnit), "The compile unit should match", exe=False,
            substrs = [os.path.join(self.mydir, 'main.c')])
        #print "compile unit:", compileUnit

        function = context.GetFunction()
        self.assertTrue(function.IsValid())
        #print "function:", function

        block = context.GetBlock()
        self.assertTrue(block.IsValid())
        #print "block:", block

        lineEntry = context.GetLineEntry()
        self.expect(lineEntry.GetFileSpec().GetDirectory(), "The line entry should have the correct directory",
                    exe=False,
            substrs = [self.mydir])
        self.expect(lineEntry.GetFileSpec().GetFilename(), "The line entry should have the correct filename",
                    exe=False,
            substrs = ['main.c'])
        self.assertTrue(lineEntry.GetLine() == self.line,
                        "The line entry's line number should match ")

        symbol = context.GetSymbol()
        self.assertTrue(function.GetName() == symbol.GetName() and symbol.GetName() == 'c',
                        "The symbol name should be 'c'")

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
