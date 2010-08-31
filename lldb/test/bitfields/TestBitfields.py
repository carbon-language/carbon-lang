"""Show bitfields and check that they display correctly."""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestBitfields(TestBase):

    mydir = "bitfields"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @unittest2.expectedFailure
    def test_with_dsym_and_run_command(self):
        """Test 'variable list ...' on a variable with bitfields."""
        self.buildDsym()
        self.bitfields_variable()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_and_python_api(self):
        """Use Python APIs to inspect a bitfields variable."""
        self.buildDsym()
        self.bitfields_variable_python()

    @unittest2.expectedFailure
    def test_with_dwarf_and_run_command(self):
        """Test 'variable list ...' on a variable with bitfields."""
        self.buildDwarf()
        self.bitfields_variable()

    def test_with_dwarf_and_python_api(self):
        """Use Python APIs to inspect a bitfields variable."""
        self.buildDwarf()
        self.bitfields_variable_python()

    def bitfields_variable(self):
        """Test 'variable list ...' on a variable with bitfields."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        self.expect("breakpoint set -f main.c -l 42", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = 42, locations = 1")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is Stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # This should display correctly.
        self.expect("variable list bits", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(uint32_t:1) b1 = 0x00000001,',
                       '(uint32_t:2) b2 = 0x00000003,',
                       '(uint32_t:3) b3 = 0x00000007,',
                       '(uint32_t:4) b4 = 0x0000000f,',
                       '(uint32_t:5) b5 = 0x0000001f,',
                       '(uint32_t:6) b6 = 0x0000003f,',
                       '(uint32_t:7) b7 = 0x0000007f,',
                       '(uint32_t:4) four = 0x0000000f'])

        # And so should this.
        # rdar://problem/8348251
        self.expect("variable list", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(uint32_t:1) b1 = 0x00000001,',
                       '(uint32_t:2) b2 = 0x00000003,',
                       '(uint32_t:3) b3 = 0x00000007,',
                       '(uint32_t:4) b4 = 0x0000000f,',
                       '(uint32_t:5) b5 = 0x0000001f,',
                       '(uint32_t:6) b6 = 0x0000003f,',
                       '(uint32_t:7) b7 = 0x0000007f,',
                       '(uint32_t:4) four = 0x0000000f'])

    def bitfields_variable_python(self):
        """Use Python APIs to inspect a bitfields variable."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.c", 42)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)

        self.runCmd("run", RUN_SUCCEEDED)
        # This does not work, and results in the process stopped at dyld_start?
        #process = target.LaunchProcess([''], [''], os.ctermid(), False)

        # The stop reason of the thread should be breakpoint.
        thread = target.GetProcess().GetThreadAtIndex(0)
        self.assertTrue(thread.GetStopReason() == StopReasonEnum("Breakpoint"),
                        STOPPED_DUE_TO_BREAKPOINT)

        # The breakpoint should have a hit count of 1.
        self.assertTrue(breakpoint.GetHitCount() == 1, BREAKPOINT_HIT_ONCE)

        # Lookup the "bits" variable which contains 8 bitfields.
        frame = thread.GetFrameAtIndex(0)
        bits = frame.LookupVar("bits")
        self.DebugSBValue(frame, bits)
        self.assertTrue(bits.GetTypeName() == "Bits" and
                        bits.GetNumChildren() == 8 and
                        bits.GetByteSize() == 4,
                        "(Bits)bits with byte size of 4 and 8 children")

        b1 = bits.GetChildAtIndex(0)
        self.DebugSBValue(frame, b1)
        self.assertTrue(b1.GetName() == "b1" and
                        b1.GetTypeName() == "uint32_t:1" and
                        b1.IsInScope(frame) and
                        int(b1.GetValue(frame), 16) == 0x01,
                        'bits.b1 has type uint32_t:1, is in scope, and == 0x01')

        b7 = bits.GetChildAtIndex(6)
        self.DebugSBValue(frame, b7)
        self.assertTrue(b7.GetName() == "b7" and
                        b7.GetTypeName() == "uint32_t:7" and
                        b7.IsInScope(frame) and
                        int(b7.GetValue(frame), 16) == 0x7f,
                        'bits.b7 has type uint32_t:7, is in scope, and == 0x7f')

        four = bits.GetChildAtIndex(7)
        self.DebugSBValue(frame, four)
        self.assertTrue(four.GetName() == "four" and
                        four.GetTypeName() == "uint32_t:4" and
                        four.IsInScope(frame) and
                        int(four.GetValue(frame), 16) == 0x0f,
                        'bits.four has type uint32_t:4, is in scope, and == 0x0f')


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
