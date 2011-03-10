"""
Test SBThread APIs.
"""

import os, time
import unittest2
import lldb
from lldbutil import get_stopped_thread, get_caller_symbol
from lldbtest import *

class ThreadAPITestCase(TestBase):

    mydir = os.path.join("python_api", "thread")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_get_stop_description_with_dsym(self):
        """Test Python SBThread.GetStopDescription() API."""
        self.buildDsym()
        self.get_stop_description()

    @python_api_test
    def test_get_stop_description_with_dwarf(self):
        """Test Python SBThread.GetStopDescription() API."""
        self.buildDwarf()
        self.get_stop_description()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_step_out_of_malloc_into_function_b_with_dsym(self):
        """Test Python SBThread.StepOut() API to step out of a malloc call where the call site is at function b()."""
        # We build a different executable than the default buildDsym() does.
        d = {'CXX_SOURCES': 'main2.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.step_out_of_malloc_into_function_b()

    @python_api_test
    def test_step_out_of_malloc_into_function_b_with_dwarf(self):
        """Test Python SBThread.StepOut() API to step out of a malloc call where the call site is at function b()."""
        # We build a different executable than the default buildDwarf() does.
        d = {'CXX_SOURCES': 'main2.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.step_out_of_malloc_into_function_b()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_step_over_3_times_with_dsym(self):
        """Test Python SBThread.StepOver() API."""
        # We build a different executable than the default buildDsym() does.
        d = {'CXX_SOURCES': 'main2.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.step_over_3_times()

    @python_api_test
    def test_step_over_3_times_with_dwarf(self):
        """Test Python SBThread.StepOver() API."""
        # We build a different executable than the default buildDwarf() does.
        d = {'CXX_SOURCES': 'main2.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.step_over_3_times()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number("main.cpp", "// Set break point at this line and check variable 'my_char'.")
        # Find the line numbers within main2.cpp for step_over_3_times() and step_out_of_malloc_into_function_b().
        self.line2 = line_number("main2.cpp", "// thread step-out of malloc into function b.")
        self.line3 = line_number("main2.cpp", "// we should reach here after 3 step-over's.")

    def get_stop_description(self):
        """Test Python SBProcess.ReadMemory() API."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)
        #self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        self.process = target.Launch (self.dbg.GetListener(), None, None, os.ctermid(), os.ctermid(), os.ctermid(), None, 0, False, error)

        thread = get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint")
        #self.runCmd("process status")

        # Due to the typemap magic (see lldb.swig), we pass in an (int)length to GetStopDescription
        # and expect to get a Python string as the result object!
        # The 100 is just an arbitrary number specifying the buffer size.
        stop_description = thread.GetStopDescription(100)
        self.expect(stop_description, exe=False,
            startstr = 'breakpoint')

    def step_out_of_malloc_into_function_b(self):
        """Test Python SBThread.StepOut() API to step out of a malloc call where the call site is at function b()."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByName('malloc')
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        self.process = target.Launch (self.dbg.GetListener(), None, None, os.ctermid(), os.ctermid(), os.ctermid(), None, 0, False, error)

        while True:
            thread = get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
            self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint")
            caller_symbol = get_caller_symbol(thread)
            #print "caller symbol of malloc:", caller_symbol
            if not caller_symbol:
                self.fail("Test failed: could not locate the caller symbol of malloc")
            if caller_symbol == "b(int)":
                break
            #self.runCmd("thread backtrace")
            #self.runCmd("process status")           
            self.process.Continue()

        thread.StepOut()
        self.runCmd("thread backtrace")
        #self.runCmd("process status")           
        self.assertTrue(thread.GetFrameAtIndex(0).GetLineEntry().GetLine() == self.line2,
                        "step out of malloc into function b is successful")

    def step_over_3_times(self):
        """Test Python SBThread.StepOver() API."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation('main2.cpp', self.line2)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        self.process = target.Launch (self.dbg.GetListener(), None, None, os.ctermid(), os.ctermid(), os.ctermid(), None, 0, False, error)

        self.assertTrue(self.process.IsValid(), PROCESS_IS_VALID)

        # Frame #0 should be on self.line2.
        self.assertTrue(self.process.GetState() == lldb.eStateStopped)
        thread = get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint condition")
        self.runCmd("thread backtrace")
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertTrue(lineEntry.GetLine() == self.line2)

        thread.StepOver()
        thread.StepOver()
        thread.StepOver()
        self.runCmd("thread backtrace")

        # Verify that we are stopped at the correct source line number in main2.cpp.
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertTrue(thread.GetStopReason() == lldb.eStopReasonPlanComplete)
        self.assertTrue(lineEntry.GetLine() == self.line3)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
