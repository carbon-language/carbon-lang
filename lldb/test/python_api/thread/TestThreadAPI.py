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
    def test_get_process_with_dsym(self):
        """Test Python SBThread.GetProcess() API."""
        self.buildDsym()
        self.get_process()

    @python_api_test
    def test_get_process_with_dwarf(self):
        """Test Python SBThread.GetProcess() API."""
        self.buildDwarf()
        self.get_process()

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
    def test_run_to_address_with_dsym(self):
        """Test Python SBThread.RunToAddress() API."""
        # We build a different executable than the default buildDwarf() does.
        d = {'CXX_SOURCES': 'main2.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.run_to_address()

    @python_api_test
    def test_run_to_address_with_dwarf(self):
        """Test Python SBThread.RunToAddress() API."""
        # We build a different executable than the default buildDwarf() does.
        d = {'CXX_SOURCES': 'main2.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.run_to_address()

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

    def get_process(self):
        """Test Python SBThread.GetProcess() API."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        self.process = target.LaunchSimple(None, None, os.getcwd())

        thread = get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint")
        self.runCmd("process status")

        proc_of_thread = thread.GetProcess()
        #print "proc_of_thread:", proc_of_thread
        self.assertTrue(proc_of_thread.GetProcessID() == self.process.GetProcessID())

    def get_stop_description(self):
        """Test Python SBThread.GetStopDescription() API."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)
        #self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        self.process = target.LaunchSimple(None, None, os.getcwd())

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

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByName('malloc')
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        self.process = target.LaunchSimple(None, None, os.getcwd())

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

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation('main2.cpp', self.line2)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        self.process = target.LaunchSimple(None, None, os.getcwd())

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

    def run_to_address(self):
        """Test Python SBThread.RunToAddress() API."""
        # We build a different executable than the default buildDwarf() does.
        d = {'CXX_SOURCES': 'main2.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)

        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation('main2.cpp', self.line2)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        self.process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(self.process.IsValid(), PROCESS_IS_VALID)

        # Frame #0 should be on self.line2.
        self.assertTrue(self.process.GetState() == lldb.eStateStopped)
        thread = get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint condition")
        self.runCmd("thread backtrace")
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertTrue(lineEntry.GetLine() == self.line2)

        # Get the start/end addresses for this line entry.
        start_addr = lineEntry.GetStartAddress().GetLoadAddress(target)
        end_addr = lineEntry.GetEndAddress().GetLoadAddress(target)
        if self.TraceOn():
            print "start addr:", hex(start_addr)
            print "end addr:", hex(end_addr)

        # Disable the breakpoint.
        self.assertTrue(target.DisableAllBreakpoints())
        self.runCmd("breakpoint list")
        
        thread.StepOver()
        thread.StepOver()
        thread.StepOver()
        self.runCmd("thread backtrace")

        # Now ask SBThread to run to the address 'start_addr' we got earlier, which
        # corresponds to self.line2 line entry's start address.
        thread.RunToAddress(start_addr)
        self.runCmd("process status")
        #self.runCmd("thread backtrace")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
