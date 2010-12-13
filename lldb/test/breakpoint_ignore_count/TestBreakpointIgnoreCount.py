"""
Test breakpoint ignore count features.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class BreakpointIgnoreCountTestCase(TestBase):

    mydir = "breakpoint_ignore_count"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_and_run_command(self):
        """Exercise breakpoint ignore count with 'breakpoint set -i <count>'."""
        self.buildDsym()
        self.breakpoint_ignore_count()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_with_dsym_and_python_api(self):
        """Use Python APIs to set breakpoint ignore count."""
        self.buildDsym()
        self.breakpoint_ignore_count_python()

    def test_with_dwarf_and_run_command(self):
        """Exercise breakpoint ignore count with 'breakpoint set -i <count>'."""
        self.buildDwarf()
        self.breakpoint_ignore_count()

    @python_api_test
    def test_with_dwarf_and_python_api(self):
        """Use Python APIs to set breakpoint ignore count."""
        self.buildDwarf()
        self.breakpoint_ignore_count_python()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line1 = line_number('main.c', '// Find the line number of function "c" here.')
        self.line2 = line_number('main.c', '// b(2) -> c(2) Find the call site of b(2).')
        self.line3 = line_number('main.c', '// a(3) -> c(3) Find the call site of c(3).')
        self.line4 = line_number('main.c', '// a(3) -> c(3) Find the call site of a(3).')

    def breakpoint_ignore_count(self):
        """Exercise breakpoint ignore count with 'breakpoint set -i <count>'."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Create a breakpoint by function name 'c'.
        self.expect("breakpoint set -i 1 -f main.c -l %d" % self.line1, BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = %d, locations = 1" %
                        self.line1)

        # Now run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # The process should be stopped at this point.
        self.expect("process status", PROCESS_STOPPED,
            patterns = ['Process .* stopped'])

        # Also check the hit count, which should be 2, due to ignore count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_THRICE,
            substrs = ["resolved = 1",
                       "hit count = 2"])

        # The frame #0 should correspond to main.c:37, the executable statement
        # in function name 'c'.  And frame #2 should point to main.c:45.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT_IGNORE_COUNT,
            #substrs = ["stop reason = breakpoint"],
            patterns = ["frame #0.*main.c:%d" % self.line1,
                        "frame #2.*main.c:%d" % self.line2])

    def breakpoint_ignore_count_python(self):
        """Use Python APIs to set breakpoint ignore count."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        self.assertTrue(breakpoint.IsValid() and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Get the breakpoint location from breakpoint after we verified that,
        # indeed, it has one location.
        location = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location.IsValid() and
                        location.IsEnabled(),
                        VALID_BREAKPOINT_LOCATION)

        # Set the ignore count on the breakpoint location.
        location.SetIgnoreCount(2)
        self.assertTrue(location.GetIgnoreCount() == 2,
                        "SetIgnoreCount() works correctly")

        # Now launch the process, and do not stop at entry point.
        self.process = target.LaunchProcess([''], [''], os.ctermid(), 0, False)

        self.process = target.GetProcess()
        self.assertTrue(self.process.IsValid(), PROCESS_IS_VALID)

        # Frame#0 should be on main.c:37, frame#1 should be on main.c:25, and
        # frame#2 should be on main.c:48.
        #lldbutil.PrintStackTraces(self.process)
        frame0 = self.process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        frame1 = self.process.GetThreadAtIndex(0).GetFrameAtIndex(1)
        frame2 = self.process.GetThreadAtIndex(0).GetFrameAtIndex(2)
        self.assertTrue(frame0.GetLineEntry().GetLine() == self.line1 and
                        frame1.GetLineEntry().GetLine() == self.line3 and
                        frame2.GetLineEntry().GetLine() == self.line4,
                        STOPPED_DUE_TO_BREAKPOINT_IGNORE_COUNT)

        # The hit count for the breakpoint should be 3.
        self.assertTrue(breakpoint.GetHitCount() == 3)

        self.process.Continue()

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
