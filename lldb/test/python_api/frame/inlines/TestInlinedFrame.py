"""
Testlldb Python SBFrame APIs IsInlined() and GetFunctionName().
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class InlinedFrameAPITestCase(TestBase):

    mydir = os.path.join("python_api", "frame", "inlines")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_stop_at_outer_inline_dsym(self):
        """Exercise SBFrame.IsInlined() and SBFrame.GetFunctionName()."""
        self.buildDsym()
        self.do_stop_at_outer_inline()

    @python_api_test
    def test_stop_at_outer_inline_with_dwarf(self):
        """Exercise SBFrame.IsInlined() and SBFrame.GetFunctionName()."""
        self.buildDwarf()
        self.do_stop_at_outer_inline()

    def do_stop_at_outer_inline(self):
        """Exercise SBFrame.IsInlined() and SBFrame.GetFunctionName()."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('outer_inline', 'a.out')
        #print "breakpoint:", breakpoint
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() > 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        process = target.GetProcess()
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        # The first breakpoint should correspond to an inlined call frame.
        frame0 = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        self.assertTrue(frame0.IsInlined() and
                        frame0.GetFunctionName() == 'outer_inline')

        self.runCmd("bt")
        if self.TraceOn():
            print "Full stack traces when first stopped on the breakpoint 'outer_inline':"
            import lldbutil
            print lldbutil.print_stacktraces(process)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
