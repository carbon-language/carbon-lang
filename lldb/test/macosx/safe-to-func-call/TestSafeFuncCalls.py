"""Test function call thread safety."""

import os, time
import unittest2
import lldb
import lldbutil
from lldbtest import *

class TestSafeFuncCalls(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_with_dsym_and_python_api(self):
        """Test function call thread safety."""
        self.buildDsym()
        self.function_call_safety_check()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dwarf_test
    def test_with_dwarf_and_python_api(self):
        """Test function call thread safety."""
        self.buildDwarf()
        self.function_call_safety_check()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = "main.c"



    def check_number_of_threads(self, process):
        self.assertTrue(process.GetNumThreads() == 2, "Check that the process has two threads when sitting at the stopper() breakpoint")

    def safe_to_call_func_on_main_thread (self, main_thread):
        self.assertTrue(main_thread.SafeToCallFunctions() == True, "It is safe to call functions on the main thread")

    def safe_to_call_func_on_select_thread (self, select_thread):
        self.assertTrue(select_thread.SafeToCallFunctions() == False, "It is not safe to call functions on the select thread")

    def function_call_safety_check(self):
        """Test function call safety checks"""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.main_source_spec = lldb.SBFileSpec (self.main_source)
        break1 = target.BreakpointCreateByName ("stopper", 'a.out')
        self.assertTrue(break1, VALID_BREAKPOINT)
        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        threads = lldbutil.get_threads_stopped_at_breakpoint (process, break1)
        if len(threads) != 1:
            self.fail ("Failed to stop at breakpoint 1.")

        self.check_number_of_threads(process)

        main_thread = lldb.SBThread()
        select_thread = lldb.SBThread()
        for idx in range (0, process.GetNumThreads()):
            t = process.GetThreadAtIndex (idx)
            if t.GetName() == "main thread":
                main_thread = t
            if t.GetName() == "select thread":
                select_thread = t

        self.assertTrue(main_thread.IsValid() and select_thread.IsValid(), "Got both expected threads")

        self.safe_to_call_func_on_main_thread (main_thread)
        self.safe_to_call_func_on_select_thread (select_thread)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
