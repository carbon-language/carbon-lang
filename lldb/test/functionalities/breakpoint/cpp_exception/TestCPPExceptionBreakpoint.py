"""
Test that you can set breakpoint and hit the C++ language exception breakpoint
"""

import os
import re
import unittest2
import lldb, lldbutil
import sys
from lldbtest import *

class TestCPPExceptionBreakpoint (TestBase):

    mydir = TestBase.compute_mydir(__file__)
    my_var = 10

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_cpp_exception_breakpoint (self):
        """Test setting and hitting the C++ exception breakpoint."""
        self.buildDsym()
        self.do_cpp_exception_bkpt ()

    @python_api_test
    @dwarf_test
    def test_cpp_exception_breakpoint_with_dwarf(self):
        """Test setting and hitting the C++ exception breakpoint."""
        self.buildDwarf()
        self.do_cpp_exception_bkpt ()

    def setUp (self):
        TestBase.setUp(self)
        self.main_source = "main.c"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)


    def do_cpp_exception_bkpt (self):
        exe = os.path.join(os.getcwd(), "a.out")
        error = lldb.SBError()

        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        exception_bkpt = self.target.BreakpointCreateForException(lldb.eLanguageTypeC_plus_plus, False, True)
        self.assertTrue (exception_bkpt.IsValid(), "Created exception breakpoint.")

        process = self.target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        
        thread_list = lldbutil.get_threads_stopped_at_breakpoint (process, exception_bkpt)
        self.assertTrue (len(thread_list) == 1, "One thread stopped at the exception breakpoint.")
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
