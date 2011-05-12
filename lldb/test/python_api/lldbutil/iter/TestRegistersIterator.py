"""
Test the iteration protocol for frame registers.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *

class RegistersIteratorTestCase(TestBase):

    mydir = "python_api/lldbutil/iter"

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line1 = line_number('main.cpp', '// Set break point at this line.')

    def test_iter_registers(self):
        """Test iterator works correctly for lldbutil.iter_registers()."""
        self.buildDefault()
        self.iter_registers()

    def iter_registers(self):
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line1)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        rc = lldb.SBError()
        self.process = target.Launch (self.dbg.GetListener(), None, None, os.ctermid(), os.ctermid(), os.ctermid(), None, 0, False, rc)

        if not rc.Success() or not self.process.IsValid():
            self.fail("SBTarget.LaunchProcess() failed")

        import lldbutil
        for thread in self.process:
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                for frame in thread:
                    # Dump the registers of this frame using iter_registers().
                    if self.TraceOn():
                        print frame

                    for kind in ["General Purpose Registers",
                                 "Floating Point Registers",
                                 "Exception State Registers"]:
                        REGs = lldbutil.get_registers(frame, kind)
                        if self.TraceOn():
                            print "%s:" % kind
                        for reg in REGs:
                            self.assertTrue(reg.IsValid())
                            if self.TraceOn():
                                print "%s => %s" % (reg.GetName(), reg.GetValue(frame))

                    # And these should also work.
                    self.assertTrue(lldbutil.get_GPRs(frame))
                    self.assertTrue(lldbutil.get_FPRs(frame))
                    self.assertTrue(lldbutil.get_ESRs(frame))
                    break


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
