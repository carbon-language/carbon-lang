"""Test stepping through ObjC method dispatch in various forms."""

import os, time
import unittest2
import lldb
from lldbtest import *

class TestObjCStepping(TestBase):

    mydir = "objc-stepping"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_with_dsym_and_python_api(self):
        """Test stepping through ObjC method dispatch in various forms."""
        self.buildDsym()
        self.objc_stepping()

    @python_api_test
    def test_with_dwarf_and_python_api(self):
        """Test stepping through ObjC method dispatch in various forms."""
        self.buildDwarf()
        self.objc_stepping()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "stepping-tests.m"
        self.line1 = line_number(self.main_source, '// Set first breakpoint here.')
        self.line2 = line_number(self.main_source, '// Set second breakpoint here.')
        self.line3 = line_number(self.main_source, '// Set third breakpoint here.')
        self.line4 = line_number(self.main_source, '// Set fourth breakpoint here.')
        self.line5 = line_number(self.main_source, '// Set fifth breakpoint here.')
        self.source_randomMethod_line = line_number (self.main_source, '// Source randomMethod start line.')
        self.sourceBase_randomMethod_line = line_number (self.main_source, '// SourceBase randomMethod start line.')
        self.source_returnsStruct_start_line = line_number (self.main_source, '// Source returnsStruct start line.')
        self.source_returnsStruct_call_line = line_number (self.main_source, '// Source returnsStruct call line.')
        self.sourceBase_returnsStruct_start_line = line_number (self.main_source, '// SourceBase returnsStruct start line.')

    def objc_stepping(self):
        """Use Python APIs to test stepping into ObjC methods."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        break1 = target.BreakpointCreateByLocation(self.main_source, self.line1)
        self.assertTrue(break1.IsValid(), VALID_BREAKPOINT)

        break2 = target.BreakpointCreateByLocation(self.main_source, self.line2)
        self.assertTrue(break2.IsValid(), VALID_BREAKPOINT)

        break3 = target.BreakpointCreateByLocation(self.main_source, self.line3)
        self.assertTrue(break3.IsValid(), VALID_BREAKPOINT)

        break4 = target.BreakpointCreateByLocation(self.main_source, self.line4)
        self.assertTrue(break4.IsValid(), VALID_BREAKPOINT)

        break5 = target.BreakpointCreateByLocation(self.main_source, self.line5)
        self.assertTrue(break5.IsValid(), VALID_BREAKPOINT)

        break_returnStruct_call_super = target.BreakpointCreateByLocation(self.main_source, self.source_returnsStruct_call_line)
        self.assertTrue(break_returnStruct_call_super.IsValid(), VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        self.process = target.LaunchProcess([], [], os.ctermid(), 0, False)

        self.assertTrue(self.process.IsValid(), PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread = self.process.GetThreadAtIndex(0)
        if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
            from lldbutil import StopReasonString
            self.fail(STOPPED_DUE_TO_BREAKPOINT_WITH_STOP_REASON_AS %
                      StopReasonString(thread.GetStopReason()))

        # Make sure we stopped at the first breakpoint.

        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertTrue (line_number == self.line1, "Hit the first breakpoint.")

        mySource = thread.GetFrameAtIndex(0).FindVariable("mySource")
        self.assertTrue(mySource.IsValid(), "Found mySource local variable.")
        mySource_isa = mySource.GetChildMemberWithName ("isa")
        self.assertTrue(mySource_isa.IsValid(), "Found mySource->isa local variable.")
        mySource_isa.GetValue (thread.GetFrameAtIndex(0))

        # Now step in, that should leave us in the Source randomMethod:
        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertTrue (line_number == self.source_randomMethod_line, "Stepped into Source randomMethod.")

        # Now step in again, through the super call, and that should leave us in the SourceBase randomMethod:
        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertTrue (line_number == self.sourceBase_randomMethod_line, "Stepped through super into SourceBase randomMethod.")

        self.process.Continue()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertTrue (line_number == self.line2, "Continued to second breakpoint in main.")

        # Again, step in twice gets us to a stret method and a stret super call:
        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertTrue (line_number == self.source_returnsStruct_start_line, "Stepped into Source returnsStruct.")

        self.process.Continue()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertTrue (line_number == self.source_returnsStruct_call_line, "Stepped to the call super line in Source returnsStruct.")

        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertTrue (line_number == self.sourceBase_returnsStruct_start_line, "Stepped through super into SourceBase returnsStruct.")

        # Cool now continue to get past the call that intializes the Observer, and then do our steps in again to see that 
        # we can find our way when we're stepping through a KVO swizzled object.

        self.process.Continue()
        frame = thread.GetFrameAtIndex(0)
        line_number = frame.GetLineEntry().GetLine()
        self.assertTrue (line_number == self.line3, "Continued to third breakpoint in main, our object should now be swizzled.")
        
        mySource_isa.GetValue (frame)
        did_change = mySource_isa.GetValueDidChange (frame)

        self.assertTrue (did_change, "The isa did indeed change, swizzled!")

        # Now step in, that should leave us in the Source randomMethod:
        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertTrue (line_number == self.source_randomMethod_line, "Stepped into Source randomMethod in swizzled object.")

        # Now step in again, through the super call, and that should leave us in the SourceBase randomMethod:
        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertTrue (line_number == self.sourceBase_randomMethod_line, "Stepped through super into SourceBase randomMethod in swizzled object.")

        self.process.Continue()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertTrue (line_number == self.line4, "Continued to fourth breakpoint in main.")

        # Again, step in twice gets us to a stret method and a stret super call:
        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertTrue (line_number == self.source_returnsStruct_start_line, "Stepped into Source returnsStruct in swizzled object.")

        self.process.Continue()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertTrue (line_number == self.source_returnsStruct_call_line, "Stepped to the call super line in Source returnsStruct - second time.")

        thread.StepInto()
        line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertTrue (line_number == self.sourceBase_returnsStruct_start_line, "Stepped through super into SourceBase returnsStruct in swizzled object.")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
